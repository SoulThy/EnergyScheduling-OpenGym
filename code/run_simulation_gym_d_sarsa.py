#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

"""
Gym-side D-SARSA agent: same learning logic as legacy Node-embedded D-SARSA,
but using SchedulingEnv to collect transitions and an external DSPSarsaTiling.

Goal: show that "legacy D-SARSA inside Node" and "D-SARSA on Gym env" achieve
similar convergence curves (validation for thesis).

Configuration is aligned with run_simulation_d_sarsa.py and sim_builder defaults.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import numpy as np

from function_approximation import DSPSarsaTiling
from gym_env import SchedulingEnv
from node import Node


# Hyperparameters aligned with sim_builder / legacy D-SARSA
EPISODE_LENGTH = 60
REWARD_ALPHA = 1.0
SIM_TIME = 10_000
TILING_NUM_TILINGS = 26
TILING_MAX_SIZE = 33_554_432
SARSA_ALPHA = 0.01
SARSA_BETA = 0.01
EPS_INIT = 0.90
EPS_DECAY = 0.9995
EPS_MIN = 0.05


class DSARSAAgent:
    """
    D-SARSA agent using DSPSarsaTiling, re-expressing the logic from Node
    (_act_d_sarsa, _d_sarsa_learn_episode) for use with SchedulingEnv.
    """

    def __init__(
        self,
        value_function: DSPSarsaTiling,
        possible_actions_fn: callable,
        *,
        eps: float = EPS_INIT,
        eps_decay: float = EPS_DECAY,
        eps_min: float = EPS_MIN,
    ) -> None:
        self._value_function = value_function
        self._possible_actions_fn = possible_actions_fn
        self._epsilon = eps
        self._eps_decay = eps_decay
        self._eps_min = eps_min

    @property
    def current_epsilon(self) -> float:
        return self._epsilon

    def _q(self, state: list[int | float], action: int) -> float:
        return self._value_function.value(state + [action])

    def select_action(self, obs: np.ndarray, possible_actions: list[int] | None = None) -> int:
        state = obs.tolist()
        if possible_actions is None:
            possible_actions = self._possible_actions_fn()
        if not possible_actions:
            return 0
        if np.random.rand() <= self._epsilon:
            return int(np.random.choice(possible_actions))
        values = [(a, self._q(state, a)) for a in possible_actions]
        max_q = max(v[1] for v in values)
        best_actions = [a for a, q in values if q == max_q]
        return int(np.random.choice(best_actions))

    def learn_episode(
        self,
        transitions: list[tuple[list[int | float], int]],
        rewards_by_step: dict[int, float],
        episode_base: int,
        alpha: float = 1.0,
    ) -> tuple[float, float]:
        """
        One batch update over an episode, matching Node._d_sarsa_learn_episode.
        transitions[i] = (s_i, a_i) for step episode_base + i.
        rewards_by_step[episode_base + i] = r_i (reward for job i when it completed).
        """
        if len(transitions) == 0:
            return 0.0, 0.0
        n = len(transitions)
        losses = []
        # First transition: (s_0, a_0) -> r_0 -> (s_1, a_1)
        if episode_base not in rewards_by_step:
            return 0.0, 0.0
        state, action = transitions[0]
        reward = rewards_by_step[episode_base]
        for i in range(1, n):
            step_idx = episode_base + i
            if step_idx not in rewards_by_step:
                return (sum(losses) / len(losses)) if losses else 0.0, 0.0
            next_state, next_action = transitions[i]
            current_full = state + [action]
            next_full = next_state + [next_action]
            loss = self._value_function.learn(current_full, next_full, reward)
            losses.append(abs(loss))
            reward = rewards_by_step[step_idx]
            state, action = next_state, next_action
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return avg_loss, 0.0

    def decay_epsilon(self) -> None:
        if self._epsilon > self._eps_min:
            self._epsilon *= self._eps_decay
            self._epsilon = max(self._epsilon, self._eps_min)


def main() -> None:
    session_id = datetime.now().strftime("%Y%m%d_gym_d_sarsa")

    env = SchedulingEnv(
        actions_space_type=Node.ActionsSpace.ONLY_WORKERS,
        state_type=Node.StateType.JOB_TYPE,
        episode_length=EPISODE_LENGTH,
        reward_alpha=REWARD_ALPHA,
        simulation_time=SIM_TIME,
        sim_time_limit=None,
        max_steps_per_episode=None,
        session_id=session_id,
    )

    value_function = DSPSarsaTiling(
        num_tilings=TILING_NUM_TILINGS,
        max_size=TILING_MAX_SIZE,
        alpha=SARSA_ALPHA,
        beta=SARSA_BETA,
    )
    agent = DSARSAAgent(
        value_function,
        env.get_possible_actions,
        eps=EPS_INIT,
        eps_decay=EPS_DECAY,
        eps_min=EPS_MIN,
    )

    obs, info = env.reset()
    global_step = 0
    episode_index = 0
    episode_return = 0.0
    episode_start_time = time.time()
    total_processed_jobs = 0

    # Episode buffers: (s, a) per step; rewards by step_index when jobs complete
    episode_transitions: list[tuple[list[Any], int]] = []
    rewards_by_step: dict[int, float] = {}
    # Pending: (episode_base, transitions, score) so we log with correct episode and score
    pending_episodes: list[tuple[int, list[tuple[list[Any], int]], float]] = []

    while env._sim_env.now < SIM_TIME:  # type: ignore[attr-defined]
        possible_actions = env.get_possible_actions()
        action = agent.select_action(obs, possible_actions=possible_actions)

        # Store (s, a) for this step (state = obs as list)
        state_list = obs.tolist()
        episode_transitions.append((state_list, action))

        obs, reward, done, truncated, info = env.step(action)
        episode_return += float(reward)

        # Collect (step_index, reward) from completed jobs
        for step_idx, r in info.get("completed", []):
            rewards_by_step[step_idx] = r

        global_step += 1

        # Every EPISODE_LENGTH steps, close logical episode and try to learn
        if global_step % EPISODE_LENGTH == 0:
            episode_base = episode_index * EPISODE_LENGTH
            pending_episodes.append((episode_base, list(episode_transitions), episode_return))
            episode_transitions = []

            # Flush any pending episode that has all 60 rewards (in order)
            while pending_episodes:
                base, trans, score = pending_episodes[0]
                needed = set(range(base, base + len(trans)))
                if not needed.issubset(rewards_by_step.keys()):
                    break
                avg_loss, mse = agent.learn_episode(
                    trans, rewards_by_step, base, alpha=REWARD_ALPHA
                )
                for k in needed:
                    rewards_by_step.pop(k, None)
                pending_episodes.pop(0)

                total_processed_jobs += len(trans)
                flushed_episode = base // EPISODE_LENGTH
                elapsed = time.time() - episode_start_time
                average_reward = score / EPISODE_LENGTH if EPISODE_LENGTH > 0 else 0.0
                generated_jobs = total_processed_jobs
                cur_episode = flushed_episode + 1
                diff_episode = 1

                env.log_episode_summary(
                    node_uid=0,
                    episode=flushed_episode,
                    eps=agent.current_epsilon,
                    score=score,
                    jobs=EPISODE_LENGTH,
                    average_reward=average_reward,
                    processed_jobs=total_processed_jobs,
                    generated_jobs=generated_jobs,
                    cur_episode=cur_episode,
                    diff_episode=diff_episode,
                    now=env._sim_env.now,  # type: ignore[attr-defined]
                    elapsed=elapsed,
                    loss=avg_loss,
                    mse=mse,
                )
                print(
                    f"[GymDSARSA] episode={flushed_episode} return={score:.3f} "
                    f"eps={agent.current_epsilon:.3f} steps={global_step} "
                    f"env_now={env._sim_env.now:.3f}"  # type: ignore[attr-defined]
                )

            agent.decay_epsilon()
            episode_index += 1
            episode_return = 0.0
            episode_start_time = time.time()

    if getattr(env, "_data_storage", None) is not None:
        env._data_storage.done_simulation()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
