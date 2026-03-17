#!/usr/bin/env bash

# Run the EnergyScheduling simulator in a Docker container with
# probing-aware configuration parameters.
#
set -euo pipefail

docker run --platform linux/amd64 --rm -it \
  --name energysim-sim \
  --env MODEL_VERSION=LEGACY \
  --env MAX_PARALLEL_SIMULATIONS=2 \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  energysim bash

