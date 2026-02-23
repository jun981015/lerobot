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
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("drift")
@dataclass
class DriftConfig(PreTrainedConfig):
    """Configuration class for DriftPolicy.

    Drift is a 1-step diffusion policy that shares the same U-Net backbone as
    DiffusionPolicy but performs a single denoising step at inference time.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Action prediction horizon (total number of actions generated per call).
        n_action_steps: Number of actions actually executed in the environment per policy call.
        normalization_mapping: Maps FeatureType strings to NormalizationMode values.
        drop_n_last_frames: Number of trailing frames to drop during data loading to avoid
            excessive padding. Typically set to `horizon - n_action_steps - n_obs_steps + 1`.
        vision_backbone: torchvision ResNet variant used as the image encoder.
        crop_shape: (H, W) to crop input images to before the backbone. None = no crop.
        crop_is_random: Use random crop during training; always center-crop at eval time.
        pretrained_backbone_weights: torchvision pretrained weights identifier. None = random init.
        use_group_norm: Replace BatchNorm with GroupNorm in the backbone.
        spatial_softmax_num_keypoints: Number of keypoints for the SpatialSoftmax pooling layer.
        use_separate_rgb_encoder_per_camera: Use an independent RGB encoder for each camera view.
        down_dims: Channel widths for each U-Net downsampling stage. Also controls depth.
        kernel_size: Temporal convolution kernel size in the U-Net.
        n_groups: Number of groups for GroupNorm inside U-Net residual blocks.
        diffusion_step_embed_dim: Output dimension of the diffusion timestep embedding network.
        use_film_scale_modulation: Use FiLM scale modulation in addition to bias modulation.
        num_train_timesteps: Total number of forward-diffusion steps used during training.
        beta_schedule: Noise schedule name (e.g. "squaredcos_cap_v2").
        beta_start: Beta value at the first training timestep.
        beta_end: Beta value at the last training timestep.
        prediction_type: What the U-Net predicts — "epsilon" (noise) or "sample" (clean action).
        clip_sample: Clip denoised samples to [-clip_sample_range, +clip_sample_range].
        clip_sample_range: Magnitude of the clipping range (requires normalized action space).
        do_mask_loss_for_padding: Mask the loss over copy-padded action frames.
        optimizer_lr: Learning rate for AdamW.
        optimizer_betas: AdamW beta coefficients.
        optimizer_eps: AdamW epsilon.
        optimizer_weight_decay: AdamW weight decay.
        scheduler_name: LR scheduler type (e.g. "cosine").
        scheduler_warmup_steps: Number of warmup steps for the LR scheduler.
    """

    # Temporal structure
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Avoids excessive padding at the tail of each episode
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Vision backbone
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # U-Net architecture
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise / diffusion schedule (used during training)
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Loss
    do_mask_loss_for_padding: bool = False

    # Optimizer / scheduler presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. "
                f"Got {self.prediction_type}."
            )

        # U-Net downsamples by 2 at each stage; horizon must be divisible by total factor.
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon must be divisible by the U-Net downsampling factor "
                f"(2^len(down_dims)). Got horizon={self.horizon} and down_dims={self.down_dims}."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("At least one image input or the environment state must be provided.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` {self.crop_shape} must fit within the image shape "
                        f"{image_ft.shape} of `{key}`."
                    )

        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_ft.shape:
                    raise ValueError(
                        f"All image inputs must have the same shape. "
                        f"`{key}` {image_ft.shape} != `{first_key}` {first_ft.shape}."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
