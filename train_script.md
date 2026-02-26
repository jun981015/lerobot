# Lerobot
train script config 확인 하는 법
python src/script/lerobot_train.py or lerobot_train_robocasa.py --help
# ROBOCASA
conda activate fine

## Diffusion Policy - RoboCasa PnPC2M 학습 설정

### 환경
- GPU: A100 80GB × 4
- Dataset: `/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M`
  - 3000 에피소드, 1,281,129 프레임, 20 FPS
  - Task: PnPCounterToMicrowave

---

### Action (7D)
dim 7-10은 상수 0 (base 이동 미사용) → `action_keep_dims=7` 로 드롭

| dim | 의미 |
|-----|------|
| 0-2 | eef position (x, y, z) |
| 3-5 | eef orientation |
| 6   | gripper |
| ~~7-10~~ | ~~base 이동 (상수 0, 제거)~~ |

---

### State Keys (32D)
base 정보 및 velocity 제외

| 키 | 차원 | 비고 |
|----|------|------|
| robot0_eef_pos | 3 | end-effector 위치 |
| robot0_eef_quat | 4 | end-effector 자세 (quaternion) |
| robot0_joint_pos | 7 | joint 각도 |
| robot0_joint_pos_cos | 7 | joint 각도 cos 인코딩 |
| robot0_joint_pos_sin | 7 | joint 각도 sin 인코딩 |
| robot0_gripper_qpos | 2 | 그리퍼 위치 |
| ~~robot0_gripper_qvel~~ | ~~2~~ | ~~그리퍼 속도 (제거 추천)~~ |
| **합계** | **30** | |

제외 키: base_pos, base_quat, base_to_eef_pos, base_to_eef_quat, joint_vel, gripper_qvel

---

### 카메라 (3개, 128×128 → 112×112 random crop)

| 키 |
|----|
| observation.images.robot0_agentview_left |
| observation.images.robot0_agentview_right |
| observation.images.robot0_eye_in_hand |

---

### 모델 설정 (DiffusionConfig)

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `horizon` | `8` | 8%8=0 ✓ |
| `n_obs_steps` | `2` | 기본값 |
| `n_action_steps` | `7` | |
| `drop_n_last_frames` | `0` | 8-7-2+1=0 |
| `vision_backbone` | `"resnet18"` | |
| `use_separate_rgb_encoder_per_camera` | `True` | 카메라별 독립 encoder (33.6M) |
| `crop_shape` | `(112, 112)` | 128×128 입력에서 random crop |
| `crop_is_random` | `True` | train: random, eval: center |
| `pretrained_backbone_weights` | `None` | scratch 학습 |
| `use_group_norm` | `True` | scratch라 BN → GN |
| `spatial_softmax_num_keypoints` | `32` | 기본값 |
| `down_dims` | `(512, 1024, 2048)` | 기본값 |
| `noise_scheduler_type` | `"DDPM"` | 기본값 |
| `num_train_timesteps` | `100` | 기본값 |

### global_cond_dim 계산
```
feature_dim per camera = 64  (SpatialSoftmax 32kp × 2)
img_feat = 64 × 3 cameras = 192  (separate encoder이므로 동일)
per_step = 32 (state) + 192 (img) = 224
global_cond = 224 × 2 (n_obs_steps) = 448
FiLM cond_dim = 448 + 128 (timestep_embed) = 576
```

---

### 학습 설정 (TrainPipelineConfig / RobocasaTrainPipelineConfig)

| 파라미터 | 값 |
|---------|-----|
| `action_keep_dims` | `7` |
| `batch_size` | `64` |
| `steps` | `300000` |
| `num_workers` | `16` |
| `log_freq` | `200` (기본값) |
| `save_freq` | `50000` |
| `optimizer_lr` | `1e-4` |
| `scheduler` | cosine (기본값) |
| `scheduler_warmup_steps` | `500` (기본값) |
| 멀티 GPU | 싱글 GPU (A100 80GB × 1) |

> **주의**: `--num_workers` 미지정 시 기본값 4로 동작 → 데이터 로딩 병목 발생. 반드시 명시할 것.
> **병렬 실행 시**: 두 job 동시 실행 기준 16 workers × 2 = 32 (64코어 서버에서 안전)

### WandB 설정

| 파라미터 | 값 |
|---------|-----|
| `wandb.enable` | `True` |
| `wandb.project` | `robocasa` |
| `wandb.entity` | `junhyeong` |
| `wandb.mode` | `online` (기본값) |
| `job_name` | `diffusion` |

---

### 학습 커맨드

```bash
conda activate fine

CUDA_VISIBLE_DEVICES=1 python /home/junhyeong/workspace/lerobot/src/lerobot/scripts/lerobot_train_robocasa.py \
  --dataset.repo_id=kitchen_pnp/PnPC2M \
  --dataset.root=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M \
  --policy.type=diffusion \
  --policy.horizon=8 \
  --policy.n_action_steps=7 \
  --policy.drop_n_last_frames=0 \
  --policy.vision_backbone=resnet18 \
  --policy.crop_shape=[112,112] \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.down_dims=[512,1024,2048] \
  --state_filtered_keys=[robot0_base_pos,robot0_base_quat,robot0_base_to_eef_pos,robot0_base_to_eef_quat,robot0_joint_vel,robot0_gripper_qvel] \
  --action_keep_dims=7 \
  --batch_size=64 \
  --steps=300000 \
  --policy.optimizer_lr=1e-4 \
  --num_workers=16 \
  --save_freq=50000 \
  --wandb.enable=true \
  --wandb.project=robocasa \
  --wandb.entity=junhyeong \
  --job_name=diffusion \
  --output_dir=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/diffusion \
  --policy.push_to_hub=false
```

---

## Timing 프로파일링 테스트

`dl` (DataLoader) vs `pre` (preprocessor) 분리 측정용.

```bash
conda activate fine

CUDA_VISIBLE_DEVICES=0 python /home/junhyeong/workspace/lerobot/src/lerobot/scripts/lerobot_train_robocasa.py \
  --dataset.repo_id=kitchen_pnp/PnPC2M \
  --dataset.root=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M \
  --policy.type=diffusion \
  --policy.horizon=8 \
  --policy.n_action_steps=7 \
  --policy.drop_n_last_frames=0 \
  --policy.vision_backbone=resnet18 \
  --policy.crop_shape=[112,112] \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.down_dims=[512,1024,2048] \
  --state_filtered_keys=[robot0_base_pos,robot0_base_quat,robot0_base_to_eef_pos,robot0_base_to_eef_quat,robot0_joint_vel,robot0_gripper_qvel] \
  --action_keep_dims=7 \
  --batch_size=128 \
  --steps=1000 \
  --log_freq=100 \
  --num_workers=16 \
  --save_checkpoint=false \
  --wandb.enable=false \
  --output_dir=/tmp/timing_test \
  --policy.push_to_hub=false
```

---

## GR00T N1.5 - RoboCasa PnPC2M 학습 설정

Base model: `nvidia/GR00T-N1.5-3B` (3B 파라미터, Vision-Language-Action)

### 모델 설정 (GrootConfig)

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `n_obs_steps` | `1` | GR00T 기본값 |
| `chunk_size` | `16` | pretrained 가중치 고정값 (`action_horizon=16`) |
| `n_action_steps` | `16` | |
| `max_state_dim` | `64` | 30D state → zero-pad |
| `max_action_dim` | `32` | pretrained 가중치 고정값, 7D action → zero-pad |
| `image_size` | `(224, 224)` | Eagle2 vision tower 입력 |
| `tune_llm` | `False` | LLM backbone 동결 |
| `tune_visual` | `False` | Vision tower 동결 |
| `tune_projector` | `True` | Projector 학습 |
| `tune_diffusion_model` | `True` | Diffusion head 학습 |
| `lora_rank` | `0` | LoRA 미사용 |
| `use_bf16` | `True` | bf16 학습 |

### 학습 설정

| 파라미터 | 값 |
|---------|-----|
| `action_keep_dims` | `7` |
| `batch_size` | `32` |
| `steps` | `300000` |
| `optimizer_lr` | `1e-4` |
| `num_workers` | `16` |

### 학습 커맨드

```bash
conda activate fine

CUDA_VISIBLE_DEVICES=1 python /home/junhyeong/workspace/lerobot/src/lerobot/scripts/lerobot_train_robocasa.py \
  --dataset.repo_id=kitchen_pnp/PnPC2M \
  --dataset.root=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M \
  --policy.type=groot \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.use_bf16=true \
  --state_filtered_keys=[robot0_base_pos,robot0_base_quat,robot0_base_to_eef_pos,robot0_base_to_eef_quat,robot0_joint_vel,robot0_gripper_qvel] \
  --action_keep_dims=7 \
  --batch_size=32 \
  --steps=300000 \
  --policy.optimizer_lr=1e-4 \
  --num_workers=8 \
  --save_freq=50000 \
  --wandb.enable=true \
  --wandb.project=robocasa \
  --wandb.entity=junhyeong \
  --job_name=groot \
  --output_dir=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/groot \
  --policy.push_to_hub=false
```

> **참고**:
> - 이미지 입력 224×224 고정 (GR00T Eagle2 vision tower), crop 설정 불필요
> - State 30D → `max_state_dim=64`으로 zero-pad, Action 7D → `max_action_dim=32`으로 zero-pad
> - LLM/Vision tower 동결, Projector + Diffusion head만 학습 (fine-tuning 기본값)
> - LoRA 사용 시 `--policy.lora_rank=8` 추가 (메모리 절약)
> - base model 첫 실행 시 HuggingFace에서 자동 다운로드 (`nvidia/GR00T-N1.5-3B`)

---

## Flow Policy - RoboCasa PnPC2M 학습 설정

Diffusion Policy와 동일한 구조, 아래 변경사항 적용.
- `policy.type=flow` (noise scheduler 제거, `num_inference_steps` 추가)

### 학습 설정

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `batch_size` | `64` | |
| `steps` | `300000` | |
| `optimizer_lr` | `1e-4` | |
| `num_workers` | `16` | |
| `num_inference_steps` | `10` | ODE solver steps (기본값) |

### 학습 커맨드

```bash
conda activate fine

CUDA_VISIBLE_DEVICES=1 python /home/junhyeong/workspace/lerobot/src/lerobot/scripts/lerobot_train_robocasa.py \
  --dataset.repo_id=kitchen_pnp/PnPC2M \
  --dataset.root=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M \
  --policy.type=flow \
  --policy.horizon=8 \
  --policy.n_action_steps=7 \
  --policy.drop_n_last_frames=0 \
  --policy.vision_backbone=resnet18 \
  --policy.crop_shape=[112,112] \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.down_dims=[512,1024,2048] \
  --policy.num_inference_steps=10 \
  --state_filtered_keys=[robot0_base_pos,robot0_base_quat,robot0_base_to_eef_pos,robot0_base_to_eef_quat,robot0_joint_vel,robot0_gripper_qvel] \
  --action_keep_dims=7 \
  --batch_size=64 \
  --steps=300000 \
  --policy.optimizer_lr=1e-4 \
  --num_workers=16 \
  --save_freq=50000 \
  --wandb.enable=true \
  --wandb.project=robocasa \
  --wandb.entity=junhyeong \
  --job_name=flow \
  --output_dir=/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/flow \
  --policy.push_to_hub=false
```
