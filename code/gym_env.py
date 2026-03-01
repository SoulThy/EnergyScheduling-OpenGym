#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

"""
Gymnasium-style environment wrapper for the energy-aware scheduling simulator.

**Option A (concurrent jobs):** One step = one job arrival at the scheduler. We advance
until the next arrival (not until the dispatched job completes), matching the real-world
case where multiple jobs are in flight. Rewards are delayed (reward at step t is for the
job dispatched at t only if it completed before the next arrival, else 0).

This module provides a skeleton that maps the standard Gymnasium Env API (reset, step,
observation_space, action_space) to the existing Node/Job/SimPy backend. It documents
which Node and Job methods to call and what (if any) simulator changes are needed for
full integration.

Usage (once implemented):
    env = SchedulingEnv(actions_space_type=Node.ActionsSpace.ONLY_WORKERS, ...)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np

# Local imports (adjust if code is run from project root or as package)
try:
    from code.node import Node
    from code.sim_builder import build_simulator
except ImportError:
    from node import Node
    from sim_builder import build_simulator


# ---------------------------------------------------------------------------
# Gymnasium standard: observation_space, action_space, reset(), step()
# ---------------------------------------------------------------------------
# You should call and/or modify:
#
# **Node (scheduler)**:
#   - Node._get_state_representation(job) -> List[int]  → observation (then cast to float32)
#   - Node._get_actions() -> List[int]                   → action space size
#   - Node._get_possible_actions(job, state)              → optional: action_mask in info
#   - Node._act_execute(action, job)                     → apply action (do NOT call Node._act)
#   - Node._get_reward(job)                              → FPS-related reward (call when job.is_done())
#   - Node._get_reward_battery(action)                   → battery-related reward
#   - Node._is_episode_over(job, state)                  → terminated
#   - Node._update_state(...)                            → called internally by _clb_job_end
#
# **Job**:
#   - job.is_done()                                     → reward is ready
#   - job.is_last_of_episode()                          → episode end
#   - job.save_state_snapshot(state)                    → before _act_execute (for learning/callbacks)
#   - job.save_action(action, ...)                      → before _act_execute
#   - job.set_reward_batteries(...)                     → before _act_execute
#   - job.a_dispatched()                                → after save_action, before _act_execute
#
# **Simulator build** (shared with run_simulation_d_sarsa):
#   - simpy.Environment()
#   - Cloud(env, ...)
#   - Node(env, ..., node_type=Node.NodeType.SCHEDULER, ...) and worker Nodes
#   - ServiceDiscovery(scheduler_id, nodes, cloud)
#   - ServiceDataStorage(nodes, session_id, ...)
#   - node.set_service_discovery(discovery); node.set_service_data_storage(data_storage); node.init()
#
# **Option A — match real-world concurrency** (likely requires Node change):
#   - One Gym step = one job ARRIVAL at the scheduler (one decision). We do NOT wait for the
#     job we dispatched to complete before processing the next arrival; we advance sim until
#     the NEXT job arrives at the scheduler (next decision point). Multiple jobs can be in
#     flight (executing at workers/cloud) while we make the next decision.
#   - When a job arrives at the scheduler, do NOT call _act(); store the pending job and
#     signal the env (e.g. SimPy event). reset() / step() wait for that signal, return
#     obs = _get_state_representation(job). When step(action) is called: set_reward_batteries,
#     save_state_snapshot, save_action, a_dispatched(), _act_execute(action, job), then run
#     sim until the NEXT job arrives at the scheduler (not until current job completes).
#   - **Delayed rewards**: reward at step t is for the job we dispatched at step t only if
#     that job completed before the next job arrived; else reward = 0. So we track the
#     last dispatched (job, action) and when we reach the next arrival, check job.is_done()
#     and set reward accordingly. Algorithms may use n-step or episode returns to handle
#     delayed feedback.
# **Original D-SARSA vs Gym**: In the original code, D-SARSA takes one action per job (same
#   as Option A), but learning (weight updates) happens only after the full batch of
#   episode_length jobs completes (_can_replay_start, _d_sarsa_learn_episode). The Gym env
#   returns per-step (possibly delayed) rewards so external agents can learn online or
#   use their own batching (e.g. n-step, episode returns).
# ---------------------------------------------------------------------------


class SchedulingEnv(gym.Env):
    """
    Gymnasium environment that wraps the existing scheduler Node and SimPy simulator.

    **Option A — concurrent jobs (matches real world):** One step = one job arrival at the
    scheduler. We dispatch that job and immediately advance the sim until the NEXT job
    arrives (next decision point). We do NOT wait for the dispatched job to complete;
    multiple jobs can be in flight. Reward is delayed: at step t we return the reward
    for the job we dispatched at step t only if it completed before the next job arrived,
    else 0. The observation is the state for the current (newly arrived) job; the action
    is the index into Node._get_actions().
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        actions_space_type: Node.ActionsSpace = Node.ActionsSpace.ONLY_WORKERS,
        state_type: Node.StateType = Node.StateType.JOB_TYPE,
        episode_length: int = 60,
        reward_alpha: float = 0.5,
        sim_time_limit: float | None = None,
        max_steps_per_episode: int | None = None,
        session_id: str = "gym_session",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            actions_space_type: Maps to Node.ActionsSpace (ONLY_WORKERS, WORKERS_OR_CLOUD, etc.).
            state_type: Maps to Node.StateType (JOB_TYPE, ONLY_NUMBER).
            episode_length: Jobs per episode (Node._episode_length).
            reward_alpha: Trade-off: alpha * reward_fps + (1 - alpha) * reward_battery.
            sim_time_limit: Optional simulation time limit in seconds (for truncated).
            max_steps_per_episode: Optional step limit per episode (for truncated).
            session_id: Session identifier for logging / ServiceDataStorage.
            **kwargs: Passed to _build_simulator_components (e.g. simulation_time, worker batteries).
        """
        super().__init__()
        self._actions_space_type = actions_space_type
        self._state_type = state_type
        self._episode_length = episode_length
        self._reward_alpha = reward_alpha
        self._sim_time_limit = sim_time_limit
        self._max_steps_per_episode = max_steps_per_episode
        self._session_id = session_id
        self._kwargs = kwargs

        # Build simulator once to infer spaces (or use fixed shapes if you know them).
        # TODO: Replace with shared builder from run_simulation_d_sarsa when available.
        self._sim_env: Any = None
        self._scheduler: Node | None = None
        self._current_job: Any = None
        self._step_count = 0

        self._build_simulator_components()

        # Observation: same as Node._get_state_representation(job) -> list of ints; cast to float32.
        obs_flat = self._get_reference_observation()
        self._obs_dim = len(obs_flat)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # Action: discrete index into Node._get_actions().
        actions = self._scheduler._get_actions()
        self._action_dim = len(actions)
        self.action_space = gym.spaces.Discrete(self._action_dim)

    def _build_simulator_components(self) -> None:
        """
        Create SimPy environment, scheduler Node, worker Nodes, Cloud, ServiceDiscovery,
        and ServiceDataStorage via the shared build_simulator so legacy and Gym use identical setup.
        """
        sim_time = self._kwargs.get("simulation_time", 10_000)
        env, nodes, cloud, discovery, data_storage = build_simulator(
            sim_time=sim_time,
            session_uid=self._session_id,
            data_storage_session_id=self._session_id,
            learning_type=Node.LearningType.NO_LEARNING,
            no_learning_policy=Node.NoLearningPolicy.RANDOM,
            actions_space=self._actions_space_type,
            state_type=self._state_type,
            reward_alpha=self._reward_alpha,
            episode_length=self._episode_length,
            gym_mode=True,
        )
        self._sim_env = env
        self._scheduler = nodes[0]
        self._nodes = nodes
        self._cloud = cloud
        self._discovery = discovery
        self._data_storage = data_storage

    def _get_reference_observation(self) -> list[int]:
        """Obtain one state vector to infer observation shape. Uses current scheduler state.

        State layout (from Node._get_state_representation):
        - ONLY_NUMBER: [job_type] + [sum(loads) per worker] + [lifespan per worker]
          -> 1 + n_workers + n_workers = 1 + 2*n_workers.
        - JOB_TYPE: [job_type] + [load per (worker, job_type)] + [lifespan per worker]
          -> 1 + n_workers * (job_periodic_types + job_exponential_types) + n_workers.
        _loads_cluster has one list per worker; each list has (job_periodic_types + job_exponential_types) entries.
        """
        if self._scheduler is None:
            return []
        n_workers = 3  # matches _build_simulator_components (3 worker nodes)
        job_periodic_types = 3
        job_exponential_types = 1
        n_job_types = job_periodic_types + job_exponential_types
        if self._state_type == Node.StateType.JOB_TYPE:
            # 1 + n_workers * n_job_types + n_workers (e.g. 1 + 3*4 + 3 = 16)
            return [0] * (1 + n_workers * n_job_types + n_workers)
        else:
            # ONLY_NUMBER: 1 + n_workers + n_workers (e.g. 1 + 3 + 3 = 7)
            return [0] * (1 + 2 * n_workers)
        # TODO: Replace with a single call to self._scheduler._get_state_representation(job)
        # after advancing to first decision point (requires Gym mode / pending job in Node).

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for a new episode.

        We advance only until the first job arrives at the scheduler (first decision point),
        not the full episode. One episode = episode_length jobs (e.g. 60); each will get one step().

        Gymnasium standard: (obs, info).

        Implementation steps (to complete):
        1. If seed is not None, call self.np_random = np.random.default_rng(seed) and
           reseed any simulator RNGs (e.g. random.seed(seed)).
        2. Rebuild or thoroughly reset the SimPy environment and all nodes so episodes
           are independent (e.g. call _build_simulator_components() again, or add a
           reset hook on Node/ServiceDataStorage).
        3. Start job generation and run the SimPy environment until the first decision
           point (first job arrival at the scheduler). This requires the simulator to
           support "Gym mode" so that at arrival we do not call Node._act() but instead
           expose the job for the env (e.g. store in self._current_job and trigger an
           event the env waits on).
        4. obs = self._extract_observation(self._current_job)
        5. info = {} (optionally add action_mask from Node._get_possible_actions(job, state))
        6. return obs, info
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self._step_count = 0
        # TODO: Rebuild or reset simulator; advance until first decision point; set self._current_job.
        self._build_simulator_components()
        # Placeholder: we have no job yet because we did not run the sim. Return zero obs.
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step (Option A): apply action, run sim until NEXT job arrives at scheduler.

        We do NOT wait for the job we just dispatched to complete. Advance until the next
        decision point (next job arrival). Reward is delayed: return reward for the job we
        dispatched THIS step only if it completed before the next arrival, else 0. Track
        (last_dispatched_job, last_action) to compute this.

        Gymnasium standard: (obs, reward, terminated, truncated, info).

        Implementation steps (to complete):
        1. Validate 0 <= action < self._action_dim (raise or clip per project policy).
        2. current_job = self._current_job (must be set by reset / previous step).
        3. state = self._scheduler._get_state_representation(current_job)
        4. Prepare job for execution (same as Node._job_first_dispatching):
           - self._scheduler._get_reward_battery(action) -> pass to job.set_reward_batteries(...)
           - current_job.save_state_snapshot(state)
           - current_job.save_action(action, ...)
           - current_job.a_dispatched()
        5. self._scheduler._act_execute(action, current_job)  # do NOT call _act()
        6. Advance SimPy until the NEXT job arrives at the scheduler (next decision point),
           or episode ends. Use _advance_until_next_decision_point(). Do NOT wait for
           current_job to complete; other jobs may be dispatched by the sim during this.
        7. Delayed reward: reward = _compute_reward(current_job, action) if current_job.is_done()
           else 0.0 (job may complete in a later step; algorithms can use n-step/episode returns).
        8. terminated = (next job is last of episode) or _is_episode_over(next_job, next_state)
        9. truncated = (self._sim_time_limit and sim_env.now >= self._sim_time_limit) or
                       (self._max_steps_per_episode and self._step_count >= self._max_steps_per_episode)
        10. next_obs = self._extract_observation(next_job) if next job else zero/terminal obs
        11. info = {"success": job.is_succeed() if job.is_done() else None, ...} (optional)
        12. self._step_count += 1; self._current_job = next_job
        13. return next_obs, reward, terminated, truncated, info
        """
        self._step_count += 1
        # Placeholder returns (no real stepping yet).
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = (
            self._max_steps_per_episode is not None
            and self._step_count >= self._max_steps_per_episode
        )
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def _advance_until_next_decision_point(self) -> Any:
        """
        Run the SimPy environment until the next job arrives at the scheduler (next decision
        point). We do NOT wait for any previously dispatched job to complete; multiple jobs
        may be in flight. Matches Option A (concurrent, real-world-like behaviour).

        Returns:
            The Job instance for the next decision, or None if episode ended / truncated.

        TODO: Requires simulator support: when a job arrives at the scheduler, the Node
        must not call _act() but instead store the job (e.g. in a queue or event) and
        signal so that this method can run env.run(until=...) or wait on an event and
        then return that job. Alternatively, run in a loop: env.run(until=...) for small
        time steps and check after each whether the scheduler has a pending job (e.g. a
        new attribute Node._pending_job_for_gym set by _job_first_dispatching in Gym mode).
        """
        raise NotImplementedError(
            "Advancing to next decision point requires Node to support Gym mode (pause at "
            "job arrival instead of calling _act)."
        )

    def _extract_observation(self, job: Any) -> np.ndarray:
        """
        Build observation from the current job (state for the scheduler).

        Call Node._get_state_representation(job), then cast to np.float32 for
        Gymnasium/Stable-Baselines3. Optionally normalize (e.g. clip or scale);
        document any normalization so results are comparable with D-SARSA.
        """
        if self._scheduler is None or job is None:
            return np.zeros(self._obs_dim, dtype=np.float32)
        state = self._scheduler._get_state_representation(job)
        return np.array(state, dtype=np.float32)

    def _compute_reward(self, job: Any, action: int) -> float:
        """
        Combined reward: alpha * reward_fps + (1 - alpha) * reward_battery.

        Call only when job.is_done(). Uses Node._get_reward(job) and
        Node._get_reward_battery(action).
        """
        if self._scheduler is None or job is None:
            return 0.0
        if not job.is_done():
            raise RuntimeError("_compute_reward must be called when job.is_done() is True")
        reward_fps = self._scheduler._get_reward(job)
        reward_battery = self._scheduler._get_reward_battery(action)
        return (
            self._reward_alpha * reward_fps
            + (1.0 - self._reward_alpha) * reward_battery
        )
