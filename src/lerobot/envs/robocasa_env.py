"""RoboCasa gymnasium environment wrapper for lerobot eval pipeline.

Observation format returned (compatible with preprocess_observation()):
    {
        "pixels": {
            "robot0_agentview_left":  np.ndarray (policy_H, policy_W, 3) uint8,
            "robot0_agentview_right": np.ndarray (policy_H, policy_W, 3) uint8,
            "robot0_eye_in_hand":     np.ndarray (policy_H, policy_W, 3) uint8,
        },
        "agent_pos": np.ndarray (30,) float32,
    }

Camera resolution vs. render resolution:
    _extract_obs() uses _get_observations() at camera_height x camera_width
    (default 128x128), matching the training data resolution exactly.
    render() calls env.sim.render() directly at render_height x render_width
    (default 512x512), independent of the policy input resolution.
    This mirrors the approach used in robocasa/scripts/playback_dataset.py.

Robosuite OpenGL convention:
    Robosuite stores camera images bottom-to-top (OpenGL Y-up convention),
    so images appear upside-down when displayed. render() applies [::-1]
    (same as playback_dataset.py) to correct the orientation for saved videos.
    _extract_obs() does NOT flip, preserving the same orientation the policy
    was trained on.

preprocess_observation() will convert this to:
    observation.images.robot0_agentview_left  : (B, C, H, W) float32 [0,1]
    observation.images.robot0_agentview_right : (B, C, H, W) float32 [0,1]
    observation.images.robot0_eye_in_hand     : (B, C, H, W) float32 [0,1]
    observation.state                         : (B, 30) float32

Note: robosuite provides robot0_joint_pos_cos/sin as direct obs keys.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# State keys used during training (sorted alphabetically, matching merge_state_subkeys order).
# Filtered out: robot0_base_pos, robot0_base_quat, robot0_base_to_eef_pos,
#               robot0_base_to_eef_quat, robot0_joint_vel, robot0_gripper_qvel
DEFAULT_STATE_KEYS = [
    "robot0_eef_pos",        # 3D  - direct robosuite key
    "robot0_eef_quat",       # 4D  - direct robosuite key
    "robot0_gripper_qpos",   # 2D  - direct robosuite key
    "robot0_joint_pos",      # 7D  - direct robosuite key
    "robot0_joint_pos_cos",  # 7D  - direct robosuite key (robosuite provides this)
    "robot0_joint_pos_sin",  # 7D  - direct robosuite key (robosuite provides this)
]  # total: 30D

DEFAULT_CAMERA_NAMES = [
    "robot0_agentview_left",
    "robot0_agentview_right",
    "robot0_eye_in_hand",
]


class RobocasaGymEnv(gym.Env):
    """Gymnasium wrapper for RoboCasa environments.

    Wraps a robosuite/RoboCasa env and returns observations in the format
    expected by lerobot's eval pipeline (preprocess_observation compatible).

    Camera resolution vs. policy input:
        Cameras are captured at camera_height x camera_width (default 224x224)
        for video quality. Images are resized to policy_height x policy_width
        (default 128x128, matching training) inside _extract_obs() before
        being passed to the policy.

    Action handling:
        The policy outputs 7D actions (eef 6D + gripper 1D). The robosuite env
        expects a full action vector including base motion dims (which are 0 during
        training). This wrapper automatically pads the policy action with zeros.

    Args:
        env_name: RoboCasa task name, e.g. "PnPCounterToMicrowave".
        robots: Robot name, e.g. "PandaOmron".
        camera_names: List of camera names to render.
        camera_height: Capture height (pixels). Used for video. Default 224.
        camera_width: Capture width (pixels). Used for video. Default 224.
        policy_height: Policy input height (pixels). Must match training. Default 128.
        policy_width: Policy input width (pixels). Must match training. Default 128.
        state_keys: Ordered list of state subkeys to concatenate into agent_pos.
            Must match the filtered keys used during training.
        horizon: Maximum episode steps.
        render_mode: Gymnasium render mode.
        layout_id: Kitchen layout ID (-1 = random).
        style_id: Kitchen style ID (-1 = random).
        obj_instance_split: Object instance split ("A" = train, "B" = eval).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_name: str = "PnPCounterToMicrowave",
        robots: str = "PandaOmron",
        camera_names: list[str] | None = None,
        camera_height: int = 128,
        camera_width: int = 128,
        state_keys: list[str] | None = None,
        horizon: int = 500,
        render_mode: str | None = "rgb_array",
        layout_id: int = -1,
        style_id: int = -1,
        obj_instance_split: str | None = None,
        render_height: int = 512,
        render_width: int = 512,
    ):
        super().__init__()

        self.camera_names = camera_names or DEFAULT_CAMERA_NAMES
        self.state_keys = state_keys or DEFAULT_STATE_KEYS
        self.render_mode = render_mode
        self._max_episode_steps = horizon
        self._obj_instance_split = obj_instance_split
        self._render_height = render_height
        self._render_width = render_width

        import robocasa  # noqa: F401 - registers RoboCasa kitchen environments with robosuite
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller=None,
            robot=robots if isinstance(robots, str) else robots[0],
        )

        make_kwargs = dict(
            env_name=env_name,
            robots=robots,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera=None,
            ignore_done=False,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=self.camera_names,
            camera_heights=camera_height,
            camera_widths=camera_width,
            camera_depths=False,
            control_freq=20,
            horizon=horizon,
            layout_ids=layout_id,
            style_ids=style_id,
            translucent_robot=False,
        )
        if obj_instance_split is not None:
            make_kwargs["obj_instance_split"] = obj_instance_split

        self._env = robosuite.make(**make_kwargs)

        # Full action dim from robosuite (includes base motion dims)
        action_low, action_high = self._env.action_spec
        self._full_action_dim = len(action_low)
        # Policy outputs only the first 7 dims (eef + gripper, no base)
        self._policy_action_dim = 7

        self.action_space = spaces.Box(
            low=action_low[: self._policy_action_dim],
            high=action_high[: self._policy_action_dim],
            dtype=np.float32,
        )

        state_dim = self._compute_state_dim()
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        cam: spaces.Box(
                            low=0,
                            high=255,
                            shape=(camera_height, camera_width, 3),
                            dtype=np.uint8,
                        )
                        for cam in self.camera_names
                    }
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(state_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def _compute_state_dim(self) -> int:
        """Compute total state dimension by doing a dummy reset."""
        raw_obs = self._env.reset()
        return sum(np.array(raw_obs[k]).flatten().shape[0] for k in self.state_keys)

    def _extract_obs(self, raw_obs: dict) -> dict:
        """Convert raw robosuite obs dict to lerobot-compatible format.

        Images are taken directly from _get_observations() at the resolution
        set in robosuite.make() (camera_height x camera_width = 128x128),
        matching the training data resolution. No resize or flip needed.
        """
        pixels = {cam: raw_obs[f"{cam}_image"] for cam in self.camera_names}

        agent_pos = np.concatenate(
            [np.array(raw_obs[k]).flatten().astype(np.float32) for k in self.state_keys]
        )

        return {"pixels": pixels, "agent_pos": agent_pos}

    def _pad_action(self, action: np.ndarray) -> np.ndarray:
        """Pad 7D policy action to full robosuite action dim.

        Padding values match the original demonstration data:
          index 0-6  : policy output (eef 6D + gripper 1D)
          index 7    : torso = 0         (never used in demos)
          index 8-10 : base vel = 0      (never used in demos)
          index 11   : base_mode = -1    (arm mode; original HDF5 value confirmed)
        """
        full_action = np.zeros(self._full_action_dim, dtype=np.float64)
        full_action[: self._policy_action_dim] = action
        # base_mode: -1 = arm mode (matches original demonstration data)
        full_action[-1] = -1.0
        return full_action

    @property
    def task_description(self) -> str:
        return self._env.__class__.__name__

    @property
    def task(self) -> str:
        return self.task_description

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            import random
            np.random.seed(seed)
            random.seed(seed)
        raw_obs = self._env.reset()
        obs = self._extract_obs(raw_obs)
        return obs, {}

    def step(self, action: np.ndarray):
        full_action = self._pad_action(action)
        raw_obs, reward, done, info = self._env.step(full_action)
        obs = self._extract_obs(raw_obs)
        # Expose success for eval pipeline (info["final_info"]["is_success"])
        info["is_success"] = bool(self._env._check_success())
        return obs, float(reward), bool(done), False, info

    def render(self):
        """Return a video frame: all cameras tiled horizontally, flipped upright.

        Uses env.sim.render() directly (same as robocasa/scripts/playback_dataset.py)
        so render resolution is independent of the policy input resolution.
        [::-1] corrects the OpenGL bottom-to-top convention.
        The result is (render_height, render_width * n_cameras, 3) uint8.
        """
        if self.render_mode != "rgb_array":
            return None

        frames = [
            self._env.sim.render(
                height=self._render_height,
                width=self._render_width,
                camera_name=cam,
            )[::-1]
            for cam in self.camera_names
        ]
        return np.concatenate(frames, axis=1)

    def close(self):
        self._env.close()


if __name__ == "__main__":
    """Smoke-test: create env → reset → step → close. Use as debug entry point."""
    import sys

    for _p in [
        "/home/junhyeong/workspace/lerobot/src",
        "/home/junhyeong/workspace/robocasa",
        "/home/junhyeong/workspace/robosuite",
    ]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    print("=== RobocasaGymEnv smoke-test ===")
    env = RobocasaGymEnv(
        env_name="PnPCounterToMicrowave",
        robots="PandaOmron",
        horizon=10,
        layout_id=1,
        style_id=1,
        obj_instance_split="B",
    )
    print(f"action_space        : {env.action_space}")
    print(f"_max_episode_steps  : {env._max_episode_steps}")
    print(f"render resolution   : {env._render_height}x{env._render_width}")

    obs, info = env.reset()
    print(f"reset OK | agent_pos: {obs['agent_pos'].shape}  dtype={obs['agent_pos'].dtype}")
    for cam, img in obs["pixels"].items():
        print(f"  pixels[{cam}]: {img.shape}  dtype={img.dtype}")

    frame = env.render()
    print(f"render frame shape  : {frame.shape}")  # (224, 672, 3)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step OK  | reward={reward:.4f}  terminated={terminated}  is_success={info.get('is_success')}")

    env.close()
    print("=== Done ===")
