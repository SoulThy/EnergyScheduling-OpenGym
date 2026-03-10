## Building the Docker image

From the project root:

```shell
cd environment && docker build . --tag energysim && cd ..
```

This recreates the computational environment locally and tags it as `energysim`.

## Running simulations (with probing-aware parameters)

Use this command to start an interactive container configured with all key environment variables, including the ones relevant for probing experiments:

```shell
docker run --platform linux/amd64 --rm -it \
  --name energysim-sim \
  --env MAX_PARALLEL_SIMULATIONS=2 \
  --env WORKER_BATTERY_CAPACITIES=7,8,9 \
  --env NET_SPEED_CLIENT_SCHEDULER_MBIT=10 \
  --env NET_SPEED_SCHEDULER_WORKER_MBIT=20 \
  --env NET_SPEED_SCHEDULER_CLOUD_MBIT=50 \
  --env POWER_MAX_TRANSMISSION_W=0.5 \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  energysim bash
```

- `MAX_PARALLEL_SIMULATIONS`: how many simulations can run in parallel.
- `WORKER_BATTERY_CAPACITIES`: worker node battery capacities (Wh) as `n1,n2,n3`.
- `NET_SPEED_*_MBIT`: effective link speeds (Mbit/s) for client→scheduler, scheduler→worker, and scheduler→cloud.
- `POWER_MAX_TRANSMISSION_W`: radio transmission power (W); combined with `NET_SPEED_SCHEDULER_WORKER_MBIT` this also determines the per-probe energy cost.

Inside the container you can then run the usual scripts, for example:

```shell
python run_simulation_gym_d_sarsa.py
```
