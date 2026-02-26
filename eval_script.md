# RoboCasa Eval 실행 명령어

## 체크포인트 현황

| Policy | Checkpoint | 경로 |
|--------|-----------|------|
| diffusion | `100000` | `/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/diffusion/checkpoints/100000/pretrained_model` |
| flow | `100000` | `/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/flow/checkpoints/100000/pretrained_model` |

---

## 기본 실행

```bash
conda activate lerobot
cd /home/junhyeong/workspace/lerobot/src
```

### Diffusion Policy

```bash
python lerobot/scripts/lerobot_eval_robocasa.py \
    --policy diffusion \
    --checkpoint 100000 \
    --n_episodes 50 \
    --batch_size 2 \
    --obj_split B \
    --output_dir /home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/diffusion/eval_results
```

### Flow Matching Policy

```bash
python lerobot/scripts/lerobot_eval_robocasa.py \
    --policy flow \
    --checkpoint 100000 \
    --n_episodes 50 \
    --batch_size 2 \
    --obj_split B \
    --output_dir /home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/flow/eval_results
```

---

## 비디오 저장 포함

```bash
# diffusion
python lerobot/scripts/lerobot_eval_robocasa.py \
    --policy diffusion \
    --checkpoint 300000 \
    --n_episodes 10 \
    --batch_size 2 \
    --obj_split B \
    --save_video \
    --output_dir /home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/diffusion/eval_results/300000

# flow
python lerobot/scripts/lerobot_eval_robocasa.py \
    --policy flow \
    --checkpoint 300000 \
    --n_episodes 10 \
    --batch_size 2 \
    --obj_split B \
    --save_video \
    --output_dir /home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs/flow/eval_results/300000
```

---

## 빠른 동작 확인 (n_episodes=5, batch_size=1)

```bash
# diffusion 빠른 테스트
python lerobot/scripts/lerobot_eval_robocasa.py \
    --policy diffusion \
    --checkpoint 100000 \
    --n_episodes 5 \
    --batch_size 1 \
    --obj_split B \
    --save_video \
    --output_dir /tmp/eval_test_diffusion

# flow 빠른 테스트
python lerobot/scripts/lerobot_eval_robocasa.py \
    --policy flow \
    --checkpoint 100000 \
    --n_episodes 3 \
    --batch_size 1 \
    --obj_split B \
    --save_video \
    --output_dir ~/workspace/eval_test_flow

```

---

## 결과 파일 구조

```
{output_dir}/
├── eval_results.json        ← 전체 성공률 + per-group 통계
└── videos/                  ← (--save_video 옵션 시)
    ├── layout1_style1_0/
    │   ├── eval_episode_0.mp4
    │   └── ...
    ├── layout2_style2_0/
    └── ...
```

### eval_results.json 주요 필드

```json
{
  "overall": {
    "pc_success": 42.0,
    "avg_sum_reward": 0.84,
    "n_episodes": 250,
    "eval_s": 3600.0
  },
  "per_group": {
    "layout1_style1": { "pc_success": 50.0, "n_episodes": 50 },
    "layout2_style2": { "pc_success": 38.0, "n_episodes": 50 },
    "layout4_style4": { "pc_success": 44.0, "n_episodes": 50 },
    "layout6_style9": { "pc_success": 40.0, "n_episodes": 50 },
    "layout7_style10": { "pc_success": 38.0, "n_episodes": 50 }
  }
}
```

---

## 인자 요약

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--policy` | (필수) | `diffusion` 또는 `flow` |
| `--checkpoint` | latest | 체크포인트 스텝 — 현재 `100000` 만 존재 |
| `--n_episodes` | 50 | 레이아웃 그룹당 에피소드 수 (총 × 5 layouts) |
| `--batch_size` | 2 | SyncVectorEnv 병렬 env 수 |
| `--horizon` | 500 | 에피소드당 최대 스텝 |
| `--obj_split` | `B` | `B`=논문 기준(unseen), `A`=학습 split |
| `--save_video` | off | 그룹당 최대 10개 비디오 저장 |
| `--use_amp` | off | `torch.autocast` 사용 (추론 속도 향상) |
| `--output_dir` | `outputs/eval_robocasa` | 결과 저장 디렉토리 |
