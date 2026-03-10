#!/usr/bin/env bash

# Run the EnergyScheduling simulator in a Docker container with
# probing-aware configuration parameters.

set -euo pipefail

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

