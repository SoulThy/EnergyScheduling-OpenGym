This [Code Ocean](https://codeocean.com) Compute Capsule will allow you to reproduce the results published by the author on your local machine<sup>1</sup>. Follow the instructions below, or consult [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information. Don't hesitate to reach out to [Support](mailto:support@codeocean.com) if you have any questions.

<sup>1</sup> You may need access to additional hardware and/or software licenses.

# Prerequisites

- [Docker Community Edition (CE)](https://www.docker.com/community-edition)
- MATLAB/MOSEK/Stata licenses where applicable

# Instructions

## The computational environment (Docker image)

This capsule is private and its environment cannot be downloaded at this time. You will need to rebuild the environment locally.

> If there's any software requiring a license that needs to be run during the build stage, you'll need to make your license available. See [our knowledge base](https://help.codeocean.com/user-manual/sharing-and-finding-published-capsules/exporting-capsules-and-reproducing-results-on-your-local-machine) for more information.

In your terminal, navigate to the folder where you've extracted the capsule and execute the following command:
```shell
cd environment && docker build . --tag energysim; cd ..
```

> This step will recreate the environment (i.e., the Docker image) locally, fetching and installing any required dependencies in the process. If any external resources have become unavailable for any reason, the environment will fail to build.

## Running the capsule to reproduce the results

In your terminal, navigate to the folder where you've extracted the capsule and execute one of the following commands, adjusting parameters as needed:
```shell
docker run --platform linux/amd64 --rm \
  --name energysim-sim \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  energysim bash run
```

By default, the simulations are run in parallel according to the `DEFAULT_MAX_PARALLEL_SIMULATIONS` value defined in `code/config.py`. You can override this at runtime via the `MAX_PARALLEL_SIMULATIONS` environment variable that is read inside the container.
Battery Capacities for worker nodes also can be adjusted by environment variable or `code/config.py`.

To change the maximum number of parallel simulations and/or worker battery capacities, pass explicitly with `--env`:

```shell
docker run --platform linux/amd64 --rm \
  --name energysim-sim \
  --env WORKER_BATTERY_CAPACITIES=7,8,9 \
  --env MAX_PARALLEL_SIMULATIONS=4 \
  --workdir /code \
  --volume "$PWD/code":/code \
  --volume "$PWD/results":/results \
  energysim bash run
```
