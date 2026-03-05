from __future__ import annotations

"""
Minimal Gym-based smoke test to validate SchedulingEnv logging.

This script:
- Instantiates the Gymnasium wrapper SchedulingEnv in NO_LEARNING / gym_mode.
- Runs a small number of episodes with a simple random agent.
- At the end of each episode, calls env.log_episode_summary(...) so that:
  - Console logging matches the original D-SARSA episode lines.
  - Episode rows are written into ServiceDataStorage.episodes via log.db.
"""

import time
from datetime import datetime

import numpy as np

from gym_env import SchedulingEnv
from node import Node


class RandomAgent:
    """Simplest possible agent: uniform random over the discrete action space."""

    def __init__(self, action_space):
        self.action_space = action_space
        # For compatibility with DSARSA-style logging, expose a dummy epsilon.
        self.current_epsilon: float = 1.0

    def select_action(self, obs: np.ndarray) -> int:  # noqa: ARG002
        return int(self.action_space.sample())


def main() -> None:
    # Configuration aligned with legacy runs: 60 jobs per logical episode,
    # ONLY_WORKERS action space, single long simulation of SIM_TIME seconds.
    EPISODE_LENGTH = 60
    REWARD_ALPHA = 1.0  # use only FPS reward for now, like many DSARSA configs
    SIM_TIME = 10_000

    session_id = datetime.now().strftime("%Y%m%d_gym_dummy")

    env = SchedulingEnv(
        actions_space_type=Node.ActionsSpace.ONLY_WORKERS,
        state_type=Node.StateType.JOB_TYPE,
        episode_length=EPISODE_LENGTH,
        reward_alpha=REWARD_ALPHA,
        # simulation_time controls the underlying SimPy horizon and should match
        # the SIMULATION_TIME used in legacy run scripts.
        simulation_time=SIM_TIME,
        # We control the simulation horizon externally via SIM_TIME; leave
        # sim_time_limit=None so the environment itself does not truncate.
        sim_time_limit=None,
        max_steps_per_episode=None,
        session_id=session_id,
    )

    agent = RandomAgent(env.action_space)

    # Single long simulation, with logical episodes every EPISODE_LENGTH steps.
    obs, info = env.reset()
    global_step = 0
    episode_index = 0
    episode_return = 0.0
    episode_start_time = time.time()
    total_processed_jobs = 0

    # Run a single long simulation, stopping when the underlying SimPy time
    # reaches SIM_TIME, to mirror the legacy NO_LEARNING runners.
    while env._sim_env.now < SIM_TIME:  # type: ignore[attr-defined]
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_return += float(reward)
        global_step += 1

        # When we have collected EPISODE_LENGTH decisions, close a logical episode.
        if global_step % EPISODE_LENGTH == 0:
            elapsed = time.time() - episode_start_time
            total_processed_jobs += EPISODE_LENGTH
            generated_jobs = total_processed_jobs
            cur_episode = episode_index + 1
            diff_episode = 1
            average_reward = (
                episode_return / EPISODE_LENGTH if EPISODE_LENGTH > 0 else 0.0
            )

            eps = agent.current_epsilon

            env.log_episode_summary(
                node_uid=0,
                episode=episode_index,
                eps=eps,
                score=episode_return,
                jobs=EPISODE_LENGTH,
                average_reward=average_reward,
                processed_jobs=total_processed_jobs,
                generated_jobs=generated_jobs,
                cur_episode=cur_episode,
                diff_episode=diff_episode,
                now=env._sim_env.now,  # type: ignore[attr-defined]
                elapsed=elapsed,
            )

            print(
                f"[GymDummy] episode={episode_index} return={episode_return:.3f} "
                f"steps={global_step} env_now={env._sim_env.now:.3f}"  # type: ignore[attr-defined]
            )

            # Prepare next logical episode on the same continuous simulation.
            episode_index += 1
            episode_return = 0.0
            episode_start_time = time.time()

    # If simulation ended in the middle of a logical episode, you can optionally
    # log a final partial episode here if desired.

    # Flush all logs and persist a single log.db containing all episodes.
    if getattr(env, "_data_storage", None) is not None:
        env._data_storage.done_simulation()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()

