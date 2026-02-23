# Drift Policy — 구현 TODO

## 1. 학습 Loss (`DriftModel.compute_loss`)

**파일:** `modeling_drift.py` → `DriftModel.compute_loss()`

현재 `NotImplementedError`로 막혀 있음. 1-step diffusion 학습 objective를 구현해야 함.

```python
def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
    # 필수 batch keys:
    #   OBS_STATE:      (B, n_obs_steps, state_dim)
    #   OBS_IMAGES:     (B, n_obs_steps, num_cameras, C, H, W)  [optional]
    #   OBS_ENV_STATE:  (B, n_obs_steps, env_dim)               [optional]
    #   ACTION:         (B, horizon, action_dim)
    #   action_is_pad:  (B, horizon)
    #
    # 1. global_cond = self._prepare_global_conditioning(batch)
    # 2. 여기에 학습 알고리즘 구현
    raise NotImplementedError
```

---

## 2. 1-step 추론 알고리즘 (`DriftModel._one_step_denoise`)

**파일:** `modeling_drift.py` → `DriftModel._one_step_denoise()`

현재 naive하게 t = T-1 에서 단순 forward pass 후 epsilon → x0 변환하는 placeholder가 있음.
실제 1-step 알고리즘으로 교체 필요.

```python
def _one_step_denoise(self, batch_size, global_cond, noise=None) -> Tensor:
    # 현재 구현: t=T-1에서 단일 UNet forward → epsilon prediction → x0 변환
    # TODO: 실제 1-step 알고리즘으로 교체
    ...
```

---

## 3. (선택) Config 파라미터 추가/수정

**파일:** `configuration_drift.py`

알고리즘에 필요한 하이퍼파라미터가 있으면 `DriftConfig`에 추가.

현재 있는 파라미터 (diffusion에서 그대로 상속):
- `num_train_timesteps`, `beta_schedule`, `beta_start`, `beta_end`
- `prediction_type` (`"epsilon"` | `"sample"`)
- `clip_sample`, `clip_sample_range`

필요 없는 파라미터 제거, 새 파라미터 추가 자유롭게 가능.

---

## 4. (선택) 학습 스크립트 / config yaml

lerobot의 학습 실행 방식:

```bash
conda run -n fine python lerobot/scripts/train.py \
    --policy.type=drift \
    ...
```

기존 diffusion config yaml 참고:
- `lerobot/configs/policy/diffusion.yaml` (있는 경우)

---

## 현재 완성된 것 (건드릴 필요 없음)

- [x] `DriftConfig` 등록 (`"drift"`)
- [x] `DriftPolicy` — `reset()`, `select_action()`, `predict_action_chunk()`, `forward()` 뼈대
- [x] `DriftModel` — RGB encoder, UNet, noise scheduler, `_prepare_global_conditioning()`, `generate_actions()`
- [x] `make_drift_pre_post_processors` — normalize/unnormalize 파이프라인
- [x] `factory.py` 등록 — `get_policy_class`, `make_policy_config`, `make_pre_post_processors`
