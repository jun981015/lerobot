# RoboCasa Eval 구현 진행 상황

## 목표
lerobot으로 학습한 Diffusion / Flow Policy를 RoboCasa 시뮬레이터에서 eval하기

---

## 완료된 작업

### 1. 패키지 버전 수정
- `transformers 5.1.0` → `5.2.0` 업그레이드
- `huggingface_hub 0.35.3` → `1.4.1` 업그레이드
- (두 버전 간 `is_offline_mode` API 충돌 해결)

### 2. RoboCasa Gym Wrapper 작성
**파일:** `lerobot/src/lerobot/envs/robocasa_env.py`

lerobot eval 파이프라인(`lerobot_eval.py`)이 기대하는 포맷으로 obs를 반환하는 gymnasium.Env 래퍼

#### Observation 변환 흐름
```
robosuite raw obs dict
  ↓ RobocasaGymEnv._extract_obs()
{"pixels": {"robot0_agentview_left": (128,128,3) uint8, ...}, "agent_pos": (30,) float32}
  ↓ preprocess_observation()  [lerobot/envs/utils.py]
{"observation.images.robot0_agentview_left": (B,3,128,128) float32 [0,1], ..., "observation.state": (B,30) float32}
  ↓ preprocessor (normalizer)
  ↓ policy.select_action()
7D action
  ↓ _pad_action(): zeros padding
12D action (dim 7-11 = 0, base 미사용)
  ↓ robosuite.step()
```

#### State 구성 (30D, 학습과 동일한 순서)
| key | dim | 비고 |
|-----|-----|------|
| robot0_eef_pos | 3 | robosuite 직접 제공 |
| robot0_eef_quat | 4 | robosuite 직접 제공 |
| robot0_gripper_qpos | 2 | robosuite 직접 제공 |
| robot0_joint_pos | 7 | robosuite 직접 제공 |
| robot0_joint_pos_cos | 7 | robosuite 직접 제공 |
| robot0_joint_pos_sin | 7 | robosuite 직접 제공 |

> **주의:** 학습 시 `merge_state_subkeys()`는 `sorted()` 순서로 key를 concatenate함.
> 위 순서가 alphabetical sorted 순서와 일치하는지 반드시 확인할 것.

#### 주요 파라미터
- `env_name`: RoboCasa 태스크명 (e.g. `"PnPCounterToMicrowave"`)
- `robots`: `"PandaOmron"`
- `horizon`: 에피소드 최대 스텝 (e.g. `500`)
- `layout_id`, `style_id`: 주방 구성 (`-1`이면 random)

#### Eval 파이프라인 필수 속성
- `_max_episode_steps`: eval 루프 종료 조건
- `task_description`: language instruction (policy에 따라 사용)
- `info["is_success"]`: 에피소드 종료 시 성공 여부 → `final_info["is_success"]`로 집계

---

## 완료된 추가 작업

### 3. Eval 스크립트 작성 (완료)
**파일:** `lerobot/src/lerobot/scripts/lerobot_eval_robocasa.py`

#### 구조
```
load_policy()  ← lerobot_load_robocasa_policy.py
  +
make_robocasa_envs()  → {task_group: {0: SyncVectorEnv}}
  +
eval_policy_all()  ← lerobot_eval.py
  +
결과 출력 + eval_results.json 저장
```

#### 표준 Eval 프로토콜 적용 완료
- `obj_instance_split="B"` — unseen 물체 인스턴스
- `layout_and_style_ids=((1,1),(2,2),(4,4),(6,9),(7,10))` — 5가지 고정 주방 조합

#### 주요 인자
| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--policy` | (필수) | `diffusion` 또는 `flow` |
| `--checkpoint` | latest | 체크포인트 스텝 (e.g. `100000`) |
| `--n_episodes` | 50 | 레이아웃 그룹당 에피소드 수 |
| `--batch_size` | 2 | SyncVectorEnv 크기 |
| `--obj_split` | `B` | `A`=학습용, `B`=논문 기준 eval |
| `--save_video` | off | 그룹당 최대 10개 비디오 저장 |
| `--output_dir` | `outputs/eval_robocasa` | 결과 저장 경로 |

#### 사용 예시
```bash
# 기본 (diffusion, split-B, 50 episodes/layout)
python lerobot_eval_robocasa.py --policy diffusion

# 비디오 저장 포함
python lerobot_eval_robocasa.py --policy flow --checkpoint 100000 \
    --n_episodes 50 --batch_size 2 --save_video
```

### 4. RobocasaGymEnv 업데이트 (완료)
**파일:** `lerobot/src/lerobot/envs/robocasa_env.py`

#### 변경 사항
- `obj_instance_split` 파라미터 추가 → `robosuite.make()` 에 전달
- `__main__` 디버그 블록 추가 (VS Code 디버거 진입점)

### 5. VS Code 디버그 설정 추가 (완료)
**파일:** `.vscode/launch.json`

추가된 설정:
- **"Debug: robocasa_env.py (smoke-test)"** — env 생성/reset/step 단계별 디버깅
- **"Debug: lerobot_eval_robocasa.py (diffusion)"** — eval 스크립트 디버깅

두 설정 모두:
- Python: `lerobot` conda 환경
- PYTHONPATH: lerobot/src + robocasa + robosuite 자동 설정
- `justMyCode: false` (robosuite 내부 코드까지 스텝 인)

---

## 다음에 해야 할 작업

### 6. Eval 실행 및 검증
```bash
cd /home/junhyeong/workspace/lerobot/src
conda activate lerobot
python lerobot/scripts/lerobot_eval_robocasa.py --policy diffusion --n_episodes 10 --batch_size 1
```

체크 포인트:
- [ ] `make_robocasa_envs()` 각 layout/style 환경 생성 정상 동작
- [ ] `eval_policy_all()` rollout 루프 진입 확인
- [ ] `info["overall"]["pc_success"]` 값 출력
- [ ] `eval_results.json` 저장 확인

---

## 체크할 파일 목록

| 파일 | 설명 |
|------|------|
| `lerobot/src/lerobot/envs/robocasa_env.py` | 새로 만든 gym wrapper — **지금 리뷰할 것** |
| `lerobot/src/lerobot/envs/utils.py` | `preprocess_observation()` — obs 포맷 변환 |
| `lerobot/src/lerobot/scripts/lerobot_eval.py` | `rollout()`, `eval_policy_all()` — eval 루프 |
| `lerobot/src/lerobot/scripts/lerobot_load_robocasa_policy.py` | policy 로딩 |
| `lerobot/src/lerobot/utils/custom_utils.py` | `merge_state_subkeys()` — state key 정렬 순서 확인용 |

---

## 검증 완료 사항
- `RobocasaGymEnv` reset/step 정상 동작 확인
- action_space: `Box(-1, 1, (7,))` ✓
- obs agent_pos shape: `(30,)` ✓
- image shape: `(128, 128, 3)` uint8 ✓
- `info["is_success"]` 반환 ✓
