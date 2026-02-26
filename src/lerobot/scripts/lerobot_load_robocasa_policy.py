#!/usr/bin/env python
"""
Load and verify a trained policy (diffusion, flow matching, or groot) saved from RoboCasa training.

Usage:
    # Load diffusion policy (latest checkpoint)
    python lerobot_load_robocasa_policy.py --policy diffusion

    # Load flow matching policy at a specific checkpoint
    python lerobot_load_robocasa_policy.py --policy flow --checkpoint 050000

    # Load groot policy
    python lerobot_load_robocasa_policy.py --policy groot

    # Specify custom outputs root
    python lerobot_load_robocasa_policy.py --policy diffusion --outputs_root /path/to/outputs
"""

import argparse
import logging
from pathlib import Path

import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)

OUTPUTS_ROOT = Path("/home/junhyeong/data/lerobot/robocasa/kitchen_pnp/PnPC2M/outputs")
POLICY_DIRS = {
    "diffusion": OUTPUTS_ROOT / "diffusion",
    "flow": OUTPUTS_ROOT / "flow",
    "groot": OUTPUTS_ROOT / "groot",
}


def get_checkpoint_path(policy_dir: Path, checkpoint: str | None) -> Path:
    """Return path to pretrained_model dir for the given checkpoint (or latest if None)."""
    checkpoints_dir = policy_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    available = sorted(checkpoints_dir.iterdir())
    if not available:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    if checkpoint is None:
        chosen = available[-1]  # latest
        logger.info(f"No checkpoint specified, using latest: {chosen.name}")
    else:
        chosen = checkpoints_dir / checkpoint
        if not chosen.exists():
            raise FileNotFoundError(
                f"Checkpoint '{checkpoint}' not found. Available: {[d.name for d in available]}"
            )

    pretrained_model_path = chosen / "pretrained_model"
    if not pretrained_model_path.exists():
        raise FileNotFoundError(f"pretrained_model directory not found: {pretrained_model_path}")

    return pretrained_model_path


def load_policy(
    policy_type: str,
    checkpoint: str | None = None,
    device: str = "cuda",
) -> tuple[PreTrainedPolicy, object, object]:
    """
    Load a trained policy along with its pre/post-processors.

    Args:
        policy_type: "diffusion" or "flow"
        checkpoint: Checkpoint step string e.g. "050000", "100000". Uses latest if None.
        device: torch device string

    Returns:
        (policy, preprocessor, postprocessor)
    """
    if policy_type not in POLICY_DIRS:
        raise ValueError(f"Unknown policy type '{policy_type}'. Choose from: {list(POLICY_DIRS)}")

    policy_dir = POLICY_DIRS[policy_type]
    pretrained_path = get_checkpoint_path(policy_dir, checkpoint)

    logger.info(f"Loading {policy_type} policy from: {pretrained_path}")

    # Load config from saved config.json
    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=str(pretrained_path))
    config.device = device

    # Load policy weights
    policy_cls = get_policy_class(config.type)
    policy = policy_cls.from_pretrained(
        pretrained_name_or_path=str(pretrained_path),
        config=config,
    )
    policy.eval()
    logger.info(f"Policy loaded: {policy.__class__.__name__} on device={device}")

    # Load pre/post-processors (normalizer, unnormalizer, etc.)
    preprocessor_overrides = {
        "device_processor": {"device": device},
    }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(pretrained_path),
        preprocessor_overrides=preprocessor_overrides,
    )
    logger.info("Pre/post-processors loaded.")

    return policy, preprocessor, postprocessor


def print_policy_info(policy: PreTrainedPolicy) -> None:
    cfg = policy.config
    logger.info(f"  type         : {cfg.type}")
    logger.info(f"  n_obs_steps  : {cfg.n_obs_steps}")
    logger.info(f"  input_features:")
    for k, v in cfg.input_features.items():
        logger.info(f"    {k}: {v}")
    logger.info(f"  output_features:")
    for k, v in cfg.output_features.items():
        logger.info(f"    {k}: {v}")
    n_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"  # parameters : {n_params:,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a RoboCasa-trained policy for evaluation.")
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["diffusion", "flow", "groot"],
        help="Policy type to load.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint step (e.g. '050000', '100000'). Defaults to latest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the policy on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    policy, preprocessor, postprocessor = load_policy(
        policy_type=args.policy,
        checkpoint=args.checkpoint,
        device=args.device,
    )

    logger.info("=" * 60)
    logger.info("Policy info:")
    print_policy_info(policy)
    logger.info("=" * 60)
    logger.info("Policy and processors are ready for evaluation.")

    # Example: how to use in an eval loop
    # policy.reset()
    # with torch.inference_mode():
    #     obs = preprocessor(raw_observation)
    #     action = policy.select_action(obs)
    #     action = postprocessor(action)

    return policy, preprocessor, postprocessor


if __name__ == "__main__":
    main()
