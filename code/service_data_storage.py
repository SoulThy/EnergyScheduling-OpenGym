#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import os
import pickle
import sqlite3
import time
from typing import List

from config import (
    NET_SPEED_SCHEDULER_WORKER_MBIT,
    POWER_MAX_TRANSMISSION_W,
    PROBE_CROSSFACTOR_J,
    PROBE_SIZE_BYTES,
    PROBING_ENERGY_COST_WH,
    PROBING_STATE_REFRESH_EVERY_K_JOBS,
    SCORE_SIMPLE_WEIGHT_B,
    SCORE_SIMPLE_WEIGHT_F,
    SCORE_SIMPLE_WEIGHT_Q,
    WORKER_BATTERY_CAPACITIES,
    MODEL_VERSION,
)
from job import Job
from log import Log
from node import Node

MODULE = "ServiceDataStorage"

BASE_LOG_DIR = "_log"
PATH_RESULTS = "../results"
PATH_RESULTS_DATA = f"{PATH_RESULTS}/data"
os.makedirs(PATH_RESULTS_DATA, exist_ok=True)

# noinspection SqlNoDataSourceInspection
class ServiceDataStorage:

    def __init__(self, nodes: List[Node], session_id: str, learning_type, no_learning_policy, action_space):
        self._nodes = nodes
        self._n_nodes = len(nodes)
        self._session_id = session_id
        self._learning_type = learning_type

        # init dirs and db
        if learning_type == Node.LearningType.NO_LEARNING:
            self._log_dir = f"{PATH_RESULTS_DATA}/{BASE_LOG_DIR}/no-learning/{no_learning_policy.name}/{session_id}"
        else:
            self._log_dir = f"{PATH_RESULTS_DATA}/{BASE_LOG_DIR}/learning/{learning_type.name}/{action_space.name}/{session_id}"

        os.makedirs(self._log_dir, exist_ok=True)

        # LOG_DB_IN_MEMORY=1: use in-memory DB (faster, needs more RAM). Good for cloud (e.g. DigitalOcean).
        # Unset or 0: use file-based DB (lower RAM, good for local runs).
        _env_val = os.getenv("LOG_DB_IN_MEMORY", "").strip().lower()
        self._db_in_memory = _env_val in ("1", "true", "yes")
        if self._db_in_memory:
            self._db_path = None
            self._db = sqlite3.connect(":memory:")
            Log.minfo(MODULE, "Log DB: in-memory (will write to file at end of run)")
        else:
            self._db_path = os.path.join(self._log_dir, "log.db")
            if os.path.exists(self._db_path):
                os.remove(self._db_path)
            self._db = sqlite3.connect(self._db_path)
            Log.minfo(MODULE, f"Log DB: file-based {self._db_path}")
        self._db_cur = self._db.cursor()

        self._init_db()

        self._log_filename = f"{self._log_dir}/log.txt"
        self._log_meta_filename = f"{self._log_dir}/meta.txt"

        self._counter_rt_jobs_executed = [0 for _ in range(self._n_nodes)]
        self._counter_rt_jobs_executed_overdeadline = [0 for _ in range(self._n_nodes)]
        self._counter_rt_jobs_rejected = [0 for _ in range(self._n_nodes)]

        self._counter_nrt_jobs_executed = [0 for _ in range(self._n_nodes)]
        self._counter_nrt_jobs_rejected = [0 for _ in range(self._n_nodes)]

        self._rewards = [0 for _ in range(self._n_nodes)]

        self._counter_total_jobs = 0

    def _init_db(self):
        self._db_cur.execute('''CREATE TABLE round (
                                                    time real, 
                                                    worker_id integer,
                                                    battery_residual real,
                                                    variance real
                                                )''')

        self._db_cur.execute('''CREATE TABLE end_batteries (
                                                    time real, 
                                                    worker_id integer,
                                                    max_battery real
                                                )''')

        self._db_cur.execute('''CREATE TABLE episodes (
                                                    episode integer, 
                                                    node_uid integer, 
                                                    eps real, 
                                                    score real, 
                                                    total_jobs integer, 
                                                    loss real, 
                                                    mse real, 
                                                    mae real
                                                )''')

        self._db_cur.execute('''CREATE TABLE jobs (
                                                    id text, 
                                                    node_uid integer, 
                                                    episode integer, 
                                                    eps real, 
                                                    state_snapshot text,
                                                    forwarded_to_node_uid integer,
                                                    forwarded_to_cluster_uid integer,
                                                    forwarded_to_cloud integer,
                                                    action integer, 
                                                    executed integer, 
                                                    rejected integer, 
                                                    type integer, 
                                                    over_deadline integer, 
                                                    done integer, 
                                                    reward real, 
                                                    time_total real, 
                                                    time_probing real, 
                                                    time_dispatching real, 
                                                    time_queue real,
                                                    time_execution real,
                                                    time_total_execution real,
                                                    generated_at real
                                                )''')

        self._db_cur.execute('''CREATE TABLE q_values (
                                                    state text,
                                                    node_uid integer,
                                                    episode integer,
                                                    action integer,
                                                    value real,
                                                    primary key (node_uid, state, action)
                                                )''')

        self._db_cur.execute('''CREATE TABLE q_values_by_time (
                                                    time integer,
                                                    state text,
                                                    node_uid integer,
                                                    action integer,
                                                    value real,
                                                    primary key (time, node_uid, state, action)
                                                )''')
        self._db_cur.execute('''CREATE TABLE probing_energy (
                                                    node_uid integer,
                                                    energy_wh real
                                                )''')
        self._db_cur.execute('''CREATE TABLE worker_energy_breakdown (
                                                    node_uid integer primary key,
                                                    idle_wh real,
                                                    execution_wh real,
                                                    transmission_wh real
                                                )''')
        self._db_cur.execute('''CREATE TABLE sim_config (
                                                    key text primary key,
                                                    value text
                                                )''')
        self._db.commit()
        Log.minfo(MODULE, "DB init")

    def _copy_db_to_file(self):
        """Dump in-memory DB to log.db on disk. Only used when _db_in_memory is True."""
        Log.minfo(MODULE, "Copying memory db to file, please wait")
        start = time.time()

        db_path = os.path.join(self._log_dir, "log.db")
        # Overwrite any existing log.db from previous runs so that the full
        # in-memory schema can be recreated without table-name conflicts.
        if os.path.exists(db_path):
            os.remove(db_path)

        new_db = sqlite3.connect(db_path)
        query = "".join(line for line in self._db.iterdump())

        # Dump old database in the new one.
        new_db.executescript(query)
        new_db.close()

        Log.minfo(MODULE, f"Done in {time.time() - start:2f}")

    def add_line_to_log(self, line):
        logfile = open(self._log_filename, "a")
        logfile.write(line + "\n")
        logfile.close()

    def _save_models(self):
        for node in self._nodes:
            Log.minfo(MODULE, f"Saving model for Node {node.get_uid()}")
            if node.get_learning_type() == Node.LearningType.D_SARSA and node.get_type() == Node.NodeType.SCHEDULER:
                fn = node.get_value_function()
                model_f = open(f"{node.get_models_dir()}/d_sarsa.model", "wb")
                pickle.dump(fn, model_f)
                model_f.close()
            else:
                Log.minfo(MODULE, "Skipped..")

    #
    # Stat data manager
    #

    def done_episode(self, node_uid, episode, eps, score, total_jobs, loss, mse, mae):
        # noinspection SqlResolve
        self._db_cur.execute(
            f'''INSERT INTO episodes VALUES ({episode}, {node_uid}, {eps},{score}, {total_jobs}, {loss}, {mse}, {mae})''')


    def done_job(self, job: Job, reward: int):
        # noinspection SqlResolve
        self._db_cur.execute(f'''INSERT INTO jobs VALUES (
                                    "{job.get_uid()}", 
                                    {job.get_originator_node_uid()}, 
                                    {job.get_episode()}, 
                                    {job.get_eps()},
                                    "{job.get_state_snapshot_str()}",
                                    {job.get_forwarded_to_node_id()},
                                    {job.get_forwarded_to_cluster_id()},
                                    {1 if job.is_forwarded_to_cloud() else 0},
                                    {job.get_action(0)},
                                    {1 if job.is_executed() else 0}, 
                                    {1 if job.is_rejected() else 0},
                                    {job.get_type()},
                                    {1 if job.is_over_deadline() else 0},
                                    {1 if job.is_done() else 0},
                                    {reward}, 
                                    {job.get_total_time()},
                                    {job.get_probing_time()},
                                    {job.get_dispatched_time()},
                                    {job.get_queue_time()},
                                    {job.get_time_execution()},
                                    {job.get_total_time_execution()},
                                    {job.get_generated_at()})
                                ''')

        # save to files
        self._rewards[job.get_originator_node_uid()] += reward
        self._counter_total_jobs += 1

        # Periodic commit so we don't hold one huge transaction in memory. Safe because:
        # - Each done_job() is a single INSERT; no multi-statement transaction.
        # - Nothing reads from the DB during the run; final commit in done_simulation() persists the rest.
        if self._counter_total_jobs > 0 and self._counter_total_jobs % 50_000 == 0:
            self._db.commit()

        if job.get_episode() % 100 == 0:
            self.print_data(only_to_file=True)

    def log_q_value(self, state, node_uid, episode, action, value):
        # noinspection SqlResolve
        self._db_cur.execute(f'''REPLACE INTO q_values VALUES ("{state}", {node_uid}, {episode}, {action}, {value})''')

    def log_q_value_at_time(self, time: int, state, node_uid, action, value):
        # noinspection SqlResolve
        self._db_cur.execute(f'''REPLACE INTO q_values_by_time VALUES (
                                    {time}, "{state}", {node_uid}, {action}, {value})''')

    def done_simulation(self):
        # Persist static simulation configuration so it can be inspected from log.db.
        # These values are global for the run and independent of node.

        # Try to infer representative job payload sizes from the scheduler node.
        periodic_size_mb = None
        exponential_size_mb = None
        try:
            scheduler_node = next(
                node for node in self._nodes if node.get_type() == Node.NodeType.SCHEDULER
            )
        except StopIteration:
            scheduler_node = None

        periodic_durations = None
        periodic_duration_std_devs = None
        periodic_rates_fps = None
        exponential_durations = None
        exponential_duration_std_devs = None
        exponential_rates_fps = None
        machine_speeds = None
        power_idle_w = None
        power_cpu_w = None

        if scheduler_node is not None:
            try:
                periodic_sizes = getattr(scheduler_node, "_job_periodic_payload_sizes_mbytes", None)
                if periodic_sizes:
                    periodic_size_mb = sum(periodic_sizes) / float(len(periodic_sizes))
                exponential_sizes = getattr(
                    scheduler_node, "_job_exponential_payload_sizes_mbytes", None
                )
                if exponential_sizes:
                    exponential_size_mb = sum(exponential_sizes) / float(len(exponential_sizes))

                periodic_durations = getattr(scheduler_node, "_job_periodic_durations", None)
                periodic_duration_std_devs = getattr(
                    scheduler_node, "_job_periodic_duration_std_devs", None
                )
                periodic_rates_fps = getattr(scheduler_node, "_job_periodic_rates_fps", None)

                exponential_durations = getattr(scheduler_node, "_job_exponential_durations", None)
                exponential_duration_std_devs = getattr(
                    scheduler_node, "_job_exponential_duration_std_devs", None
                )
                exponential_rates_fps = getattr(scheduler_node, "_job_exponential_rates_fps", None)
            except Exception:
                periodic_size_mb = None
                exponential_size_mb = None
                periodic_durations = None
                periodic_duration_std_devs = None
                periodic_rates_fps = None
                exponential_durations = None
                exponential_duration_std_devs = None
                exponential_rates_fps = None

        try:
            machine_speeds = ",".join(str(node._machine_speed) for node in self._nodes)  # type: ignore[attr-defined]
        except Exception:
            machine_speeds = None

        try:
            # Assume homogeneous power settings across nodes; read from the first node.
            first_node = self._nodes[0]
            power_idle_w = getattr(first_node, "_power_idle_w", None)
            power_cpu_w = getattr(first_node, "_power_max_cpu_w", None)
        except Exception:
            power_idle_w = None
            power_cpu_w = None

        sim_config_values = {
            "NET_SPEED_SCHEDULER_WORKER_MBIT": str(NET_SPEED_SCHEDULER_WORKER_MBIT),
            "PROBE_SIZE_BYTES": str(PROBE_SIZE_BYTES),
            "PROBE_CROSSFACTOR_J": str(PROBE_CROSSFACTOR_J),
            "PROBING_ENERGY_COST_WH": str(PROBING_ENERGY_COST_WH),
            "PROBING_STATE_REFRESH_EVERY_K_JOBS": str(PROBING_STATE_REFRESH_EVERY_K_JOBS),
            "SCORE_SIMPLE_WEIGHT_Q": str(SCORE_SIMPLE_WEIGHT_Q),
            "SCORE_SIMPLE_WEIGHT_B": str(SCORE_SIMPLE_WEIGHT_B),
            "SCORE_SIMPLE_WEIGHT_F": str(SCORE_SIMPLE_WEIGHT_F),
            "POWER_MAX_TRANSMISSION_W": str(POWER_MAX_TRANSMISSION_W),
            "WORKER_BATTERY_CAPACITIES": ",".join(str(v) for v in WORKER_BATTERY_CAPACITIES),
            # Tag to distinguish different workload / environment models in analysis.
            "MODEL_VERSION": MODEL_VERSION,
        }
        if periodic_size_mb is not None:
            sim_config_values["JOB_PERIODIC_PAYLOAD_SIZES_MB"] = ",".join(
                str(v) for v in periodic_sizes  # type: ignore[name-defined]
            )
            sim_config_values["JOB_PERIODIC_PAYLOAD_SIZE_MB"] = str(periodic_size_mb)
        if exponential_size_mb is not None:
            sim_config_values["JOB_EXPONENTIAL_PAYLOAD_SIZES_MB"] = ",".join(
                str(v) for v in exponential_sizes  # type: ignore[name-defined]
            )
            sim_config_values["JOB_EXPONENTIAL_PAYLOAD_SIZE_MB"] = str(exponential_size_mb)
        if periodic_durations is not None:
            sim_config_values["JOB_PERIODIC_DURATIONS_S"] = ",".join(str(v) for v in periodic_durations)
        if periodic_duration_std_devs is not None:
            sim_config_values["JOB_PERIODIC_DURATION_STD_DEVS_S"] = ",".join(
                str(v) for v in periodic_duration_std_devs
            )
        if periodic_rates_fps is not None:
            sim_config_values["JOB_PERIODIC_RATES_FPS"] = ",".join(str(v) for v in periodic_rates_fps)
        if exponential_durations is not None:
            sim_config_values["JOB_EXPONENTIAL_DURATIONS_S"] = ",".join(str(v) for v in exponential_durations)
        if exponential_duration_std_devs is not None:
            sim_config_values["JOB_EXPONENTIAL_DURATION_STD_DEVS_S"] = ",".join(
                str(v) for v in exponential_duration_std_devs
            )
        if exponential_rates_fps is not None:
            sim_config_values["JOB_EXPONENTIAL_RATES_FPS"] = ",".join(str(v) for v in exponential_rates_fps)
        if machine_speeds is not None:
            sim_config_values["NODE_MACHINE_SPEEDS"] = machine_speeds
        if power_idle_w is not None:
            sim_config_values["POWER_IDLE_W"] = str(power_idle_w)
        if power_cpu_w is not None:
            sim_config_values["POWER_MAX_CPU_W"] = str(power_cpu_w)

        for key, value in sim_config_values.items():
            self._db_cur.execute(
                '''INSERT OR REPLACE INTO sim_config (key, value) VALUES (?, ?)''',
                (key, value),
            )

        # Before dumping the in-memory DB to disk, persist total probing energy
        # and energy breakdown (idle / execution / transmission) per node.
        for node in self._nodes:
            try:
                energy_wh = node.get_total_probing_energy_wh()
            except AttributeError:
                energy_wh = 0.0
            self.log_probing_energy(node.get_uid(), energy_wh)
            # Only workers have batteries and contribute to worker energy shares.
            if node.get_type() != Node.NodeType.WORKER:
                continue
            try:
                idle_wh = node.get_total_idle_energy_wh()
                execution_wh = node.get_total_execution_energy_wh()
                transmission_wh = node.get_total_transmission_energy_wh()
                self.log_worker_energy_breakdown(
                    node.get_uid(), idle_wh, execution_wh, transmission_wh
                )
            except AttributeError:
                pass

        self._db.commit()
        if self._db_in_memory:
            self._copy_db_to_file()
        self._save_models()
        self._db.close()

    def print_data(self, only_to_file=False):
        total_rt_executed = sum(self._counter_rt_jobs_executed)
        total_rt_executed_overdeadline = sum(self._counter_rt_jobs_executed_overdeadline)
        total_rt_rejected = sum(self._counter_rt_jobs_rejected)
        total_nrt_executed = sum(self._counter_nrt_jobs_executed)
        total_nrt_rejected = sum(self._counter_nrt_jobs_rejected)
        total_reward = sum(self._rewards)
        total_rejected = total_rt_rejected + total_nrt_rejected

        log_meta_file = open(self._log_meta_filename, "w")
        print(self._counter_total_jobs, file=log_meta_file)
        print("total_rt_executed=%d" % total_rt_executed, file=log_meta_file)
        print("total_rt_executed_overdeadline=%d" % total_rt_executed_overdeadline, file=log_meta_file)
        if total_rt_executed > 0:
            print("total_rt_executed_overdeadline_perc=%.2f" % (total_rt_executed_overdeadline / total_rt_executed),
                  file=log_meta_file)
        print("total_rt_rejected=%d" % total_rt_rejected, file=log_meta_file)
        print("total_nrt_executed=%d" % total_nrt_executed, file=log_meta_file)
        print("total_nrt_rejected=%d" % total_nrt_rejected, file=log_meta_file)
        print("total_jobs=%d" % self._counter_total_jobs, file=log_meta_file)
        print("total_rejected=%d" % total_rejected, file=log_meta_file)
        print("total_reward=%d" % total_reward, file=log_meta_file)
        if self._counter_total_jobs > 0:
            print("total_rejected_perc=%.2f" % (total_rejected / self._counter_total_jobs), file=log_meta_file)
            print("total_reward / total_jobs=%.2f" % (total_reward / self._counter_total_jobs), file=log_meta_file)
        log_meta_file.close()

        if not only_to_file:
            print()
            print("### DataStorage report ###")
            print("total_rt_executed=%d" % total_rt_executed)
            print("total_rt_executed_overdeadline=%d" % total_rt_executed_overdeadline)
            print("total_rt_executed_overdeadline_perc=%.2f" % (total_rt_executed_overdeadline / total_rt_executed))
            print("total_rt_rejected=%d" % total_rt_rejected)
            print("total_nrt_executed=%d" % total_nrt_executed)
            print("total_nrt_rejected=%d" % total_nrt_rejected)
            print("self._counter_total_jobs=%d" % self._counter_total_jobs)
            print("total_rejected=%d" % total_rejected)
            print("total_rejected_perc=%.2f" % (total_rejected / self._counter_total_jobs))

    def get_log_dir(self):
        return self._log_dir

    def log_battery(self, timestamp, worker_id, battery_residual,variance):
        self._db_cur.execute(
            f'''INSERT INTO round VALUES (
                        {timestamp}, 
                        {worker_id},
                        {battery_residual},
                        {variance}
            )''')

    def log_end_battery(self, timestamp, worker_id, max_battery):
        self._db_cur.execute(
            f'''INSERT INTO end_batteries VALUES (
                        {timestamp}, 
                        {worker_id},
                        {max_battery}
            )''')

    def log_probing_energy(self, node_uid: int, energy_wh: float) -> None:
        """Persist total probing energy (Wh) for a node at the end of the simulation."""
        self._db_cur.execute(
            f'''INSERT INTO probing_energy VALUES (
                        {node_uid},
                        {energy_wh}
            )''')

    def log_worker_energy_breakdown(
        self, node_uid: int, idle_wh: float, execution_wh: float, transmission_wh: float
    ) -> None:
        """Persist per-node energy breakdown (idle / execution / transmission) in Wh."""
        self._db_cur.execute(
            '''INSERT OR REPLACE INTO worker_energy_breakdown
               (node_uid, idle_wh, execution_wh, transmission_wh) VALUES (?, ?, ?, ?)''',
            (node_uid, idle_wh, execution_wh, transmission_wh),
        )
