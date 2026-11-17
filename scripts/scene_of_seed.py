"""Print the kitchen (layout_id, style_id) that a RoboCasa reset seed produces.
   python scripts/scene_of_seed.py <Task> <seed> [<seed> ...]"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
if __name__ == "__main__":
    import gymnasium as gym
    import robocasa  # noqa: F401
    task, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    env = gym.make(f"robocasa/{task}", split="pretrain", seed=seeds[0])
    for s in seeds:
        env.reset(seed=s)
        k = env.unwrapped.env
        print(f"seed {s}: layout {k.layout_id} style {k.style_id}", flush=True)
