# Diffusion Policy 모델 구조 및 입출력 흐름

> 기준 코드: `src/lerobot/policies/diffusion/`
> RoboCasa PnPC2M 데이터셋 기준으로 실제 텐서 shape을 명시

---

## 전체 흐름 요약

```
[데이터셋]
  observation.images.*   (B, n_obs=2, 3_cams, 3, 128, 128)
  observation.state      (B, n_obs=2, state_dim)
  action                 (B, horizon=16, 11)
        ↓
[전처리]
  이미지: crop → 정규화 (MEAN_STD)
  state : MIN_MAX 정규화
  action: MIN_MAX 정규화
        ↓
[DiffusionPolicy.forward()]
        ↓
[DiffusionRgbEncoder]   이미지 → 1D 벡터
        ↓
[_prepare_global_conditioning]   state + image feature 합치기 → global_cond
        ↓
[DiffusionConditionalUnet1d]   noisy action + global_cond → denoised action
        ↓
  loss = MSE(pred, target)   (prediction_type="epsilon" → 노이즈 예측)
```

---

## 1. 데이터 입력 구조

### 배치 딕셔너리 (학습 시)
| 키 | shape | 설명 |
|----|-------|------|
| `observation.images` | `(B, 2, 3, 3, 128, 128)` | 스택된 카메라 (n_obs=2, n_cams=3) |
| `observation.state` | `(B, 2, state_dim)` | n_obs=2 타임스텝의 로봇 state |
| `action` | `(B, 16, 11)` | horizon=16 개의 액션 시퀀스 |
| `action_is_pad` | `(B, 16)` | 에피소드 끝에서 패딩된 액션 마스크 |

> `observation.images`는 `forward()`에서 각 카메라 키를 `torch.stack`으로 합쳐서 만듦
> (`modeling_diffusion.py:144-145`)

### 시간 인덱스 규칙 (delta_indices)
- `observation_delta_indices`: `[-1, 0]` → 현재 + 1스텝 전 관측
- `action_delta_indices`: `[-1, 0, 1, ..., 14]` → horizon=16개 액션

---

## 2. 이미지 인코더: `DiffusionRgbEncoder`

```
입력: (B, 3, H, W)   ← B = batch * n_obs_steps * n_cameras

  1) [crop]
     train: RandomCrop(crop_shape)   → (B, 3, 84, 84)  ← 기본값
     eval : CenterCrop(crop_shape)   → (B, 3, 84, 84)

  2) [backbone]
     ResNet18 [:-2]                  → (B, 512, 3, 3)   ← 84x84 기준
     (마지막 avgpool, fc 제거)
     BatchNorm → GroupNorm 교체 (use_group_norm=True 시)

  3) [SpatialSoftmax]
     (B, 512, 3, 3) → 32 keypoints   → (B, 32, 2)
     → flatten                        → (B, 64)

  4) [Linear + ReLU]
                                      → (B, 64)   ← feature_dim

출력: (B, 64)
```

**핵심 파라미터:**
| 파라미터 | 기본값 | 영향 |
|---------|--------|------|
| `vision_backbone` | `"resnet18"` | backbone 크기 (resnet34/50도 가능) |
| `crop_shape` | `(84, 84)` | 128×128에서 43% 잘림 → ⚠️ 검토 필요 |
| `pretrained_backbone_weights` | `None` | scratch 학습 (ImageNet 가중치 사용 가능) |
| `spatial_softmax_num_keypoints` | `32` | 출력 feature_dim = 32×2 = **64** |
| `use_separate_rgb_encoder_per_camera` | `False` | 3 카메라가 동일 encoder 공유 |

---

## 3. 글로벌 컨디셔닝: `_prepare_global_conditioning`

3개 카메라와 state를 합쳐 UNet에 넣을 단일 컨디셔닝 벡터를 만듦.

```
[이미지 처리]
  (B, 2, 3, 3, 128, 128)
  → rearrange: (B*2*3, 3, 128, 128)     ← batch + n_obs + n_cams 합치기
  → DiffusionRgbEncoder                 → (B*2*3, 64)
  → rearrange: (B, 2, 3*64=192)         ← 카메라 feature 이어 붙이기

[state]
  (B, 2, state_dim)

[cat + flatten]
  cat([state, img_feats], dim=-1)       → (B, 2, state_dim + 192)
  flatten(start_dim=1)                  → (B, 2 * (state_dim + 192))

예: state_dim=16  → global_cond = (B, 2*(16+192)) = (B, 416)
예: state_dim=53  → global_cond = (B, 2*(53+192)) = (B, 490)
```

---

## 4. 1D UNet: `DiffusionConditionalUnet1d`

### 타임스텝 임베딩
```
timestep scalar (B,)
  → SinusoidalPosEmb(128)    → (B, 128)
  → Linear(128, 512) + Mish
  → Linear(512, 128)          → (B, 128)   ← diffusion_step_embed_dim
```

### FiLM 컨디셔닝 벡터
```
global_feature = cat([timestep_embed, global_cond], dim=-1)
예: (B, 128 + 490) = (B, 618)   ← UNet 모든 ResBlock에 주입
```

### UNet 구조 (down_dims=(512, 1024, 2048) 기준)

```
입력: (B, 16, 11)  → rearrange → (B, 11, 16)   ← (batch, channels, time)

[Encoder]
  ResBlock(11→512)  + ResBlock(512→512)  + Downsample(stride=2)  → (B, 512,  8)
  ResBlock(512→1024)+ ResBlock(1024→1024)+ Downsample(stride=2)  → (B, 1024, 4)
  ResBlock(1024→2048)+ResBlock(2048→2048)+ Identity(마지막=no down)→ (B, 2048, 4)

[Bottleneck]
  ResBlock(2048→2048) × 2                                          → (B, 2048, 4)

[Decoder] (skip connection으로 Encoder 출력과 cat)
  cat+ResBlock(2048*2→1024)+ResBlock(1024→1024)+Upsample           → (B, 1024, 8)
  cat+ResBlock(1024*2→512) +ResBlock(512→512)  +Identity(마지막)   → (B, 512, 16)

[final_conv]
  Conv1dBlock(512→512) + Conv1d(512→11)                            → (B, 11, 16)
  → rearrange                                                       → (B, 16, 11)

출력: (B, 16, 11)   ← horizon × action_dim
```

**각 ResBlock 내부 (FiLM 모듈):**
```
x: (B, C, T)
  Conv1d → GroupNorm → Mish
  FiLM: Linear(cond_dim, C*2) → scale, bias
  out = scale * out + bias          ← use_film_scale_modulation=True
  Conv1d → GroupNorm → Mish
  + residual conv
```

---

## 5. 학습 시 손실 계산

```python
# 1) 랜덤 노이즈 생성
eps = randn_like(action)   # (B, 16, 11)

# 2) 랜덤 timestep 샘플
t ~ Uniform(0, num_train_timesteps=100)

# 3) Forward diffusion: clean action에 노이즈 추가
noisy_action = noise_scheduler.add_noise(action, eps, t)

# 4) UNet으로 노이즈 예측
pred = unet(noisy_action, t, global_cond)

# 5) MSE loss (prediction_type="epsilon": 노이즈 예측)
loss = MSE(pred, eps)

# 6) 패딩 영역 마스킹 (do_mask_loss_for_padding=False가 기본)
loss = loss * ~action_is_pad
```

---

## 6. 추론 시 액션 생성

```
noise = randn(B, 16, 11)   ← 순수 가우시안

DDPM 역방향 (t=99 → t=0, 100 스텝):
  for t in [99, 98, ..., 0]:
      pred_noise = unet(x_t, t, global_cond)
      x_{t-1}   = denoise_step(x_t, pred_noise, t)
      x_{t-1}   = clip(x_{t-1}, -1, 1)   ← clip_sample=True

최종 action_chunk: (B, 16, 11)
실제 실행 액션: action_chunk[:, 1:9, :]   ← start=n_obs-1=1, end=1+n_action_steps=9
```

> **DDIM**으로 바꾸면 추론을 100→10~20 스텝으로 줄일 수 있음 (`noise_scheduler_type="DDIM"`, `num_inference_steps=10`)

---

## 7. RoboCasa 학습 전 진단 체크리스트

### ✅ 필수 확인

| 항목 | 기본값 | RoboCasa 상황 | 권장 |
|------|--------|--------------|------|
| `crop_shape` | `(84, 84)` | 128×128에서 43% 잘림. 공간 정보 손실 위험 | `(112, 112)` 또는 `null` |
| `pretrained_backbone_weights` | `None` (scratch) | 데이터 3000ep로 scratch는 느림 | `"IMAGENET1K_V1"` 고려 (단, GroupNorm과 충돌 → use_group_norm=False 필요) |
| state key | — | 53D 전부 or 선택 | 중복/노이즈 많은 키 제거 권장 (velocity류) |
| `down_dims` | `(512, 1024, 2048)` | 가장 큰 설정. 메모리 주의 | 우선 유지, OOM 나면 `(256, 512, 1024)` |
| `horizon % 2^len(down_dims)` | — | 16 % 8 = 0 ✓ | — |
| `n_groups` | `8` | down_dims 모두 8로 나눠짐 ✓ | — |
| `clip_sample_range` | `1.0` | action이 MIN_MAX 정규화되면 [-1,1]이므로 OK | — |

### ⚠️ pretrained_backbone_weights와 use_group_norm 충돌

```python
# 이 조합은 에러 남 (modeling_diffusion.py:468-471)
pretrained_backbone_weights = "IMAGENET1K_V1"
use_group_norm = True   # ← ValueError 발생

# pretrained 쓰려면:
pretrained_backbone_weights = "IMAGENET1K_V1"
use_group_norm = False   # BatchNorm 유지
```

### 📌 state key 선택 가이드

```
전체 53D:
  robot0_base_pos        (3)   ← 베이스 위치 (고정 환경이면 노이즈)
  robot0_base_quat       (4)   ← 베이스 자세
  robot0_eef_pos         (3)   ← end-effector 위치 ★중요
  robot0_eef_quat        (4)   ← end-effector 자세 ★중요
  robot0_joint_pos       (7)   ★중요
  robot0_joint_vel       (7)   (velocity는 노이즈 많음, 제거 고려)
  robot0_gripper_qpos    (2)   ★중요
  robot0_gripper_qvel    (2)   (제거 고려)
  robot0_joint_pos_cos   (7)   joint_pos의 cos 인코딩 (중복 가능)
  robot0_joint_pos_sin   (7)   joint_pos의 sin 인코딩 (중복 가능)
  robot0_base_to_eef_pos (3)   ← 상대 위치, 유용할 수 있음
  robot0_base_to_eef_quat(4)   ← 상대 자세

최소 추천 (16D):
  joint_pos(7) + eef_pos(3) + eef_quat(4) + gripper_qpos(2)

중간 추천 (32D):
  위 16D + joint_vel(7) + base_to_eef_pos(3) + base_to_eef_quat(4) + gripper_qvel(2)
```

### 🔢 global_cond_dim 계산 (설정에 따라)

```
feature_dim = spatial_softmax_num_keypoints * 2 = 32 * 2 = 64
img_feat    = feature_dim * n_cameras = 64 * 3 = 192
per_step    = state_dim + img_feat

state_dim=16 → per_step=208 → global_cond = 208*2 = 416 → FiLM cond = 416+128 = 544
state_dim=32 → per_step=224 → global_cond = 224*2 = 448 → FiLM cond = 448+128 = 576
state_dim=53 → per_step=245 → global_cond = 245*2 = 490 → FiLM cond = 490+128 = 618
```

---

## 8. 파일 위치 정리

| 파일 | 역할 |
|------|------|
| `policies/diffusion/configuration_diffusion.py` | 모든 하이퍼파라미터 정의 |
| `policies/diffusion/modeling_diffusion.py` | 모델 구현 (DiffusionPolicy, DiffusionModel, UNet 등) |
| `policies/diffusion/processor_diffusion.py` | 정규화/역정규화 전처리 파이프라인 |
| `scripts/lerobot_train_robocasa.py` | RoboCasa 전용 학습 스크립트 (state merge 포함) |
| `utils/custom_utils.py` | `merge_state_subkeys`, `make_concat_state_collate_fn` |
