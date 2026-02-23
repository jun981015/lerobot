import torch
import gymnasium as gym
from pathlib import Path
from lerobot.policies.flow.modeling_flow import FlowPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.configs import PushtEnv
from lerobot.envs.factory import make_env
from lerobot.utils.utils import init_logging
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.processor.pipeline import PolicyProcessorPipeline

def main():
    init_logging()
    
    # Configuration
    policy_path = Path("outputs/train/example_pusht_flow")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_episodes = 10
    batch_size = 10
    
    # Load Policy
    print(f"Loading policy from {policy_path}")
    policy = FlowPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()
    
    # Create Environment
    print("Creating environment")
    env_cfg = PushtEnv()
    envs = make_env(env_cfg, n_envs=batch_size)
    
    # Create Processors
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, 
        pretrained_path=policy_path
    )
    
    # Environment specific processors (identity for PushT)
    env_preprocessor = PolicyProcessorPipeline([])
    env_postprocessor = PolicyProcessorPipeline([])
    
    # Evaluate
    print("Starting evaluation...")
    info = eval_policy_all(
        envs=envs,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=n_episodes,
        videos_dir=policy_path / "eval_videos",
        max_parallel_tasks=1,
        start_seed=0  # Fix for zip error
    )
    
    print("Evaluation results:")
    print(info["overall"])
    print(f"Videos saved to {policy_path / 'eval_videos'}")

if __name__ == "__main__":
    main()
