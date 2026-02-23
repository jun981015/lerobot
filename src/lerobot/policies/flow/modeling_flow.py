#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flow Matching Policy
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.policies.flow.configuration_flow import FlowConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.utils import get_safe_dtype


class FlowPolicy(PreTrainedPolicy):
    """
    Flow Matching Policy
    """

    config_class = FlowConfig
    name = "flow"

    def __init__(
        self,
        config: FlowConfig,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.flow_model = FlowModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.flow_model.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.flow_model.generate_actions(batch)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.flow_model.compute_loss(batch)
        return loss, None


class FlowModel(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config

        # Build observation encoders
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [FlowRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FlowRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = FlowConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

    def create_flow_matching_time(self, time, min_period=4e-3, max_period=4.0, device=None):
        dtype = get_dtype_from_parameters(self)
        dtype = get_safe_dtype(dtype, device)
        fraction = torch.linspace(0.0, 1.0, self.config.diffusion_step_embed_dim // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time[:, None]
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        return pos_emb

    def denoise_step(self, global_cond, x_t, time):
        # Embed time for flow matching
        time_emb = self.create_flow_matching_time(time, device=x_t.device)
        pred = self.unet(x_t, time_emb, global_cond=global_cond)
        return pred

    def flow_matching_action_sample(self, x_t, batch_size, global_cond):
        device = global_cond.device
        # We integrate from t=1 (noise) to t=0 (data)
        # So dt should be negative
        dt = -1.0 / self.config.num_inference_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Euler integration
        # TODO: Support other solvers like RK4
        for _ in range(self.config.num_inference_steps):
            expanded_time = time.expand(batch_size)
            v_t = self.denoise_step(
                global_cond,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time = time + dt
            
        return x_t

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond = self._prepare_global_conditioning(batch)
        
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        
        # Sample prior (Gaussian noise)
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
        )
        
        actions = self.flow_matching_action_sample(sample, batch_size, global_cond)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def sample_time(self, bsize, device):
        if self.config.use_beta_for_time:
            # Sample t from Beta distribution, biased towards edges 0 and 1
            beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0) # NOTE: Original code used 1.5, 1.0? Usually uniform [0,1] is fine for FM
            time_step = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
            # Avoid exact 0 or 1 for numerical stability if needed, though FM is usually fine
            # time = time_beta * 0.999 + 0.001 
        else:
            time_step = torch.rand(size=(bsize,), device=device, dtype=torch.float32)
        return time_step

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        # Input validation
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        
        global_cond = self._prepare_global_conditioning(batch)
        
        # Flow Matching Loss
        # x1 = data (action trajectory)
        # x0 = noise
        # t in [0, 1]
        # path: xt = t * x0 + (1-t) * x1  <-- NOTE: This matches your implementation: t=1 is noise, t=0 is data
        # velocity field u_t should target dx_t/dt = x0 - x1
        
        trajectory = batch["action"]
        batch_size = trajectory.shape[0]
        device = trajectory.device
        
        time = self.sample_time(batch_size, device)
        noise = torch.randn_like(trajectory)
        
        # Interpolation: t=1 -> noise, t=0 -> data
        # x_t = time * noise + (1 - time) * data
        # Note: broadcasting time to (B, 1, 1)
        t_expand = time[:, None, None]
        x_t = t_expand * noise + (1 - t_expand) * trajectory
        
        # Target velocity: d(x_t)/dt = noise - data
        u_t = noise - trajectory
        
        # Predict velocity
        time_emb = self.create_flow_matching_time(time, device=device)
        pred = self.unet(x_t, time_emb, global_cond=global_cond)
        
        loss = F.mse_loss(pred, u_t, reduction="none")
        
        if self.config.do_mask_loss_for_padding:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
            
        return loss.mean()


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class FlowRgbEncoder(nn.Module):
    def __init__(self, config: FlowConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


class FlowConv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class FlowConditionalUnet1d(nn.Module):
    def __init__(self, config: FlowConfig, global_cond_dim: int):
        super().__init__()
        self.config = config
        
        # Time embedding (MLP)
        self.diffusion_step_encoder = nn.Sequential(
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FlowConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        FlowConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                FlowConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                FlowConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        FlowConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        FlowConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            FlowConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, time_emb: Tensor, global_cond=None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")
        
        # In Flow Matching, we pass the time embedding directly
        timesteps_embed = self.diffusion_step_encoder(time_emb)

        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class FlowConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = FlowConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = FlowConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
