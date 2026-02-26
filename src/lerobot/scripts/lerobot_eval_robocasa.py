#!/usr/bin/env python
"""Evaluate a trained lerobot policy on RoboCasa environments.

Follows the standard RoboCasa evaluation protocol:
  - obj_instance_split="B"  (unseen object instances; training uses "A")
  - 5 fixed kitchen layout/style combos: (1,1) (2,2) (4,4) (6,9) (7,10)

The envs dict passed to eval_policy_all has shape:
    {task_group: {task_id: SyncVectorEnv}}
where task_group = "layout{L}_style{S}" and task_id = 0.

Usage:
    python lerobot_eval_robocasa.py --policy diffusion
    python lerobot_eval_robocasa.py --policy flow --checkpoint 100000 \\
        --n_episodes 50 --batch_size 2 --save_video
"""

import argparse
import json
import logging
import sys
from contextlib import nullcontext
from pathlib import Path

# Ensure packages are importable without explicit conda/PYTHONPATH setup.
for _p in [
    "/home/junhyeong/workspace/lerobot/src",
    "/home/junhyeong/workspace/robocasa",
    "/home/junhyeong/workspace/robosuite",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium as gym
import torch

from lerobot.envs.robocasa_env import RobocasaGymEnv
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.lerobot_load_robocasa_policy import load_policy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Standard RoboCasa eval protocol (robocasa paper) ──────────────────────────
LAYOUT_AND_STYLE_IDS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 2),
    (4, 4),
    (6, 9),
    (7, 10),
)
DEFAULT_ENV_NAME = "PnPCounterToMicrowave"
DEFAULT_ROBOTS = "PandaOmron"


# ── Env factory ───────────────────────────────────────────────────────────────

def make_robocasa_envs(
    env_name: str,
    robots: str,
    batch_size: int,
    horizon: int,
    obj_instance_split: str = "B",
    layout_and_style_ids: tuple[tuple[int, int], ...] = LAYOUT_AND_STYLE_IDS,
    camera_height: int = 128,
    camera_width: int = 128,
) -> dict[str, dict[int, gym.vector.SyncVectorEnv]]:
    """Return {task_group: {0: SyncVectorEnv}} for eval_policy_all.

    One SyncVectorEnv (batch_size parallel copies) is created per layout/style
    combination. All envs share the same obj_instance_split.
    """
    envs: dict[str, dict[int, gym.vector.SyncVectorEnv]] = {}

    for layout_id, style_id in layout_and_style_ids:
        task_group = f"layout{layout_id}_style{style_id}"
        logger.info(
            f"Building env: {task_group}  split={obj_instance_split}  "
            f"batch={batch_size}  horizon={horizon}"
        )

        # Capture loop variables explicitly to avoid late-binding closure bug.
        def _make_single(lid: int = layout_id, sid: int = style_id):
            return RobocasaGymEnv(
                env_name=env_name,
                robots=robots,
                camera_height=camera_height,
                camera_width=camera_width,
                horizon=horizon,
                layout_id=lid,
                style_id=sid,
                obj_instance_split=obj_instance_split,
            )

        vec_env = gym.vector.SyncVectorEnv(
            [_make_single] * batch_size,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        envs[task_group] = {0: vec_env}

    return envs


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a lerobot policy on RoboCasa (standard eval protocol)."
    )
    p.add_argument(
        "--policy",
        required=True,
        choices=["diffusion", "flow", "groot"],
        help="Policy type to load.",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint step string (e.g. '100000'). Defaults to latest.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--n_episodes",
        type=int,
        default=50,
        help="Episodes to run per layout/style group.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of parallel envs per group (SyncVectorEnv size).",
    )
    p.add_argument("--horizon", type=int, default=500, help="Max steps per episode.")
    p.add_argument("--env_name", default=DEFAULT_ENV_NAME)
    p.add_argument("--robots", default=DEFAULT_ROBOTS)
    p.add_argument(
        "--obj_split",
        default="B",
        choices=["A", "B"],
        help="Object instance split. 'B' = unseen (standard eval), 'A' = training split.",
    )
    p.add_argument(
        "--output_dir",
        default="outputs/eval_robocasa",
        help="Directory for results JSON and optional videos.",
    )
    p.add_argument(
        "--save_video",
        action="store_true",
        help="Save up to 10 rollout videos per group.",
    )
    p.add_argument(
        "--use_amp",
        action="store_true",
        help="Use torch.autocast for faster inference.",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos" if args.save_video else None

    # 1. Load policy + pre/post-processors
    logger.info(
        f"Loading {args.policy} policy  checkpoint={args.checkpoint}  device={args.device}"
    )
    policy, preprocessor, postprocessor = load_policy(
        policy_type=args.policy,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    # 2. Build envs dict
    envs = make_robocasa_envs(
        env_name=args.env_name,
        robots=args.robots,
        batch_size=args.batch_size,
        horizon=args.horizon,
        obj_instance_split=args.obj_split,
    )

    # 3. Identity env-level processors (RoboCasa needs no special env preprocessing)
    env_preprocessor: PolicyProcessorPipeline = PolicyProcessorPipeline(steps=[])
    env_postprocessor: PolicyProcessorPipeline = PolicyProcessorPipeline(steps=[])

    # 4. Run eval
    device = torch.device(args.device)
    amp_ctx = torch.autocast(device_type=device.type) if args.use_amp else nullcontext()

    logger.info(
        f"Starting eval  n_episodes={args.n_episodes}  obj_split={args.obj_split}  "
        f"layouts={list(envs.keys())}"
    )
    with torch.no_grad(), amp_ctx:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=args.n_episodes,
            max_episodes_rendered=10 if args.save_video else 0,
            videos_dir=videos_dir,
            start_seed=42,
        )

    # 5. Print results
    logger.info("=" * 60)
    logger.info("Overall results:")
    overall = info["overall"]
    logger.info(
        f"  pc_success   : {overall['pc_success']:.1f}%  "
        f"(n_episodes={overall['n_episodes']})"
    )
    logger.info(f"  avg_sum_reward: {overall['avg_sum_reward']:.4f}")
    logger.info(f"  eval_s        : {overall['eval_s']:.1f}s")
    logger.info("-" * 60)
    for group, stats in info["per_group"].items():
        logger.info(
            f"  {group:25s}  success={stats['pc_success']:.1f}%  "
            f"n={stats['n_episodes']}"
        )
    logger.info("=" * 60)

    # 6. Save results JSON
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Results saved → {results_path}")

    # 7. Close all vec envs
    for group in envs.values():
        for vec_env in group.values():
            vec_env.close()


if __name__ == "__main__":
    main()
