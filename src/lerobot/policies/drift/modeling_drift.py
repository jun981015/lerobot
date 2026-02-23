#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Drift Policy — 1-step diffusion policy.

Architecture follows DiffusionPolicy (ResNet encoder + 1D U-Net) but denoises
in a single forward pass at inference time.

Algorithm / loss implementation: TODO (fill in DriftModel.compute_loss and
DriftModel.generate_actions as needed).
"""

import math
from collections import deque
from collections.abc import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.policies.drift.configuration_drift import DriftConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------


class DriftPolicy(PreTrainedPolicy):
    """1-step diffusion policy (Drift).

    Shares the same encoder + U-Net backbone as DiffusionPolicy but performs
    a single denoising pass at inference time.
    """

    config_class = DriftConfig
    name = "drift"

    def __init__(self, config: DriftConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Queues populated during rollout; hold the n latest observations and
        # the pre-computed action chunk that is being consumed.
        self._queues = None

        self.model = DriftModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation and action queues. Call on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict an action chunk given the current observation queue."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.model.generate_actions(batch, noise=noise)
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action for the current environment step.

        Observation history is maintained in `_queues`. When the action queue
        is empty a new chunk is generated and buffered; actions are then popped
        one at a time until the buffer is exhausted.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Compute training/validation loss."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.model.compute_loss(batch)
        return loss, None


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class DriftModel(nn.Module):
    """Encoder + 1D U-Net backbone for the Drift policy."""

    def __init__(self, config: DriftConfig):
        super().__init__()
        self.config = config

        # ---- Observation encoders ----------------------------------------
        global_cond_dim = config.robot_state_feature.shape[0]

        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DriftRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DriftRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        # ---- 1D U-Net ----------------------------------------------------
        self.unet = DriftConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )

        # ---- Noise scheduler (training) ----------------------------------
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

    # ------------------------------------------------------------------
    # Observation encoding
    # ------------------------------------------------------------------

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode all observations into a flat conditioning vector (B, global_cond_dim)."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_cam = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_feats = torch.cat(
                    [enc(imgs) for enc, imgs in zip(self.rgb_encoder, images_per_cam, strict=True)]
                )
                img_feats = einops.rearrange(
                    img_feats, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_feats = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_feats = einops.rearrange(
                    img_feats, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            feats.append(img_feats)

        if self.config.env_state_feature:
            feats.append(batch[OBS_ENV_STATE])

        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Generate an action chunk from observations.

        Args:
            batch: dict with keys OBS_STATE (and optionally OBS_IMAGES /
                OBS_ENV_STATE), each of shape (B, n_obs_steps, ...).
            noise: Optional initial noise tensor of shape
                (B, horizon, action_dim). Sampled from N(0,I) if None.

        Returns:
            actions: (B, n_action_steps, action_dim)
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # TODO: implement 1-step denoising / your inference algorithm here.
        actions = self._one_step_denoise(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def _one_step_denoise(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Single-step denoising pass.

        Override or extend this method with your specific 1-step algorithm.
        By default it runs the U-Net at the highest noise level (t = T-1) and
        directly predicts the clean action trajectory.
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
            )
        )

        # Single forward pass at t = num_train_timesteps - 1.
        t = torch.full((batch_size,), self.config.num_train_timesteps - 1, dtype=torch.long, device=device)

        # TODO: replace with your 1-step denoising algorithm.
        pred = self.unet(sample, t, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            # Convert noise prediction → clean sample via the scheduler's formula.
            alpha_prod_t = self.noise_scheduler.alphas_cumprod[t].view(-1, 1, 1).to(dtype)
            sample = (sample - (1 - alpha_prod_t).sqrt() * pred) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":
            sample = pred
        else:
            raise ValueError(f"Unsupported prediction_type: {self.config.prediction_type}")

        if self.config.clip_sample:
            sample = sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        return sample

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute the training loss.

        Expected batch keys:
            OBS_STATE:      (B, n_obs_steps, state_dim)
            OBS_IMAGES:     (B, n_obs_steps, num_cameras, C, H, W)  [optional]
            OBS_ENV_STATE:  (B, n_obs_steps, env_dim)               [optional]
            ACTION:         (B, horizon, action_dim)
            action_is_pad:  (B, horizon)

        TODO: implement your training objective here.
        """
        noise = torch.randn_like(self.get_target(batch))
        x = self.unet(noise, t, global_cond=global_cond)
        y_neg = x
        y_pos = batch[ACTION]
        V = self.compute_drift(x, y_pos, y_neg)
        x_drifted = (x + V).detach()
        loss = F.mse_loss(x, x_drifted)
        return loss
    
    def compute_drift(self, x, y_pos, y_neg) -> Tensor:
        """Get the target action from the batch.
        x : [N,D]
        y_pos : [N_pos,D]
        y_neg : [N_neg,D]
        """
        dist_pos = cdist(x, y_pos) #[N,N_pos]
        dist_neg = cdist(x, y_neg) #[N,N_neg]
        if y_neg is x:
            dist_neg += torch.eye(x.shape[0]) * 1e-6
        logit_pos = -dist_pos/self.temperature
        logit_neg = -dist_neg/self.temperature
        logit = torch.cat([logit_pos, logit_neg], dim=1)
        A_row = logit.softmax(dim=-1)
        A_col = logit.softmax(dim=-2)
        A = torch.sqrt(A_row * A_col)
        A_pos, A_neg = torch.split(A,[N_pos,],dim=1)

        W_pos = A_pos #[N, N_pos]
        W_neg = A_neg #[N, N_neg]
        W_pos *=A_neg.sum(dim=1,keepdim=True)
        W_neg *=A_pos.sum(dim=1,keepdim=True)
        V_pos = W_pos @ y_pos
        V_neg = W_neg @ y_neg
        V = V_pos - V_neg

        raise NotImplementedError(
            "DriftModel.compute_drift is not yet implemented. "
            "Fill in your drift computation here."
        )
        return V        
# ---------------------------------------------------------------------------
# Backbone components
# ---------------------------------------------------------------------------


class SpatialSoftmax(nn.Module):
    """Spatial Soft Argmax — extracts 2-D keypoints from a feature map."""

    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None):
        super().__init__()
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class DriftRgbEncoder(nn.Module):
    """ResNet image encoder with optional crop and SpatialSoftmax pooling."""

    def __init__(self, config: DriftConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.crop_shape)
                if config.crop_is_random
                else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "Cannot replace BatchNorm in a pretrained model without corrupting weights."
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


class DriftSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DriftConv1dBlock(nn.Module):
    """Conv1d → GroupNorm → Mish."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DriftConditionalResidualBlock1d(nn.Module):
    """ResNet-style 1D residual block with FiLM conditioning."""

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

        self.conv1 = DriftConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DriftConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale, bias = cond_embed.chunk(2, dim=1)
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DriftConditionalUnet1d(nn.Module):
    """1D U-Net with FiLM conditioning (same topology as DiffusionPolicy U-Net)."""

    def __init__(self, config: DriftConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.diffusion_step_encoder = nn.Sequential(
            DriftSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    DriftConditionalResidualBlock1d(dim_in, dim_out, **common_kwargs),
                    DriftConditionalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                    nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.mid_modules = nn.ModuleList([
            DriftConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_kwargs),
            DriftConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common_kwargs),
        ])

        self.up_modules = nn.ModuleList()
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    DriftConditionalResidualBlock1d(dim_in * 2, dim_out, **common_kwargs),
                    DriftConditionalResidualBlock1d(dim_out, dim_out, **common_kwargs),
                    nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                ])
            )

        self.final_conv = nn.Sequential(
            DriftConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        """
        Args:
            x:           (B, T, action_dim)
            timestep:    (B,) diffusion timestep indices
            global_cond: (B, global_cond_dim)
        Returns:
            (B, T, action_dim) U-Net prediction
        """
        x = einops.rearrange(x, "b t d -> b d t")

        t_emb = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([t_emb, global_cond], dim=-1) if global_cond is not None else t_emb

        skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skip_features.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parents, k in replace_list:
        parent = root_module.get_submodule(".".join(parents)) if parents else root_module
        if isinstance(parent, nn.Sequential):
            parent[int(k)] = func(parent[int(k)])
        else:
            setattr(parent, k, func(getattr(parent, k)))
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module
