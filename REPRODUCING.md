## Building the Docker image

From the project root:

```shell
cd environment && docker build . --tag energysim && cd ..
```

This recreates the computational environment locally and tags it as `energysim`.

## Running simulations (reproducible workflow)

### Select the workload model (jobs)

We use a single environment variable to select the **job/workload model**:

- `MODEL_VERSION=LEGACY`: original (baseline) workload.
- `MODEL_VERSION=SMALL_JOBS_V1`: alternative workload preset (smaller/more frequent jobs). Values live in
  `code/sim_builder.py` under `SMALL_JOBS_V1_JOB_PARAMS` and can be edited.
- `MODEL_VERSION=SMALL_JOBS_V2`: even more frequent arrivals (approx `x4` vs `LEGACY` arrival rates), keeping the
  "SMALL" payload sizes and compute durations. Values live in `code/sim_builder.py` under `SMALL_JOBS_V2_JOB_PARAMS`.

The chosen model version and the full job parameters (payload sizes, deadlines, durations, std devs, FPS targets)
are written into each run's `log.db` under the `sim_config` table by `ServiceDataStorage.done_simulation()`.

### Start the container (minimal, safe defaults)

Use this command to start an interactive container with the baseline hardware defaults and a selected workload model.
Edit the script with the values of your liking.

```shell
./run_docker_probing.sh
```

### Optional: override hardware/network parameters

If you need to override the **hardware/platform side** (batteries, link speeds, radio power), you can still pass the
existing env vars. This is optional and more error-prone, so prefer defaults unless you are intentionally exploring:

- `WORKER_BATTERY_CAPACITIES` (Wh): `n1,n2,n3`
- `NET_SPEED_CLIENT_SCHEDULER_MBIT`, `NET_SPEED_SCHEDULER_WORKER_MBIT`, `NET_SPEED_SCHEDULER_CLOUD_MBIT` (Mbit/s)
- `POWER_MAX_TRANSMISSION_W` (W)
- `PROBE_SIZE_BYTES` (bytes) and `PROBING_ENERGY_COST_WH` (Wh), if you want to control probing directly
- `LOG_DB_IN_MEMORY=1`: keep the log database in RAM during the run (faster, needs more RAM). Good for cloud (e.g. DigitalOcean). When set, the DB is still written to `log.db` on disk at the end. Default: file-based DB (lower RAM, for local runs).

All of these are also written into `log.db` (`sim_config`) for reproducibility.

Inside the container you can then run the usual scripts, for example:

```shell
python run_simulation_gym_d_sarsa.py
```
