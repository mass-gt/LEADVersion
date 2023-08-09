# Connector tour-2-network

## Installation

The `requirements.txt` and `Pipenv` files are provided for the setup of an environment where the module can be installed. The package includes a `setup.py` file and it can be therefore installed with a `pip install .` when we are at the same working directory as the `setup.py` file. For testing purposes, one can also install the package in editable mode `pip install -e .`.

After the install is completed, an executable will be available to the user.

Furthermore, a `Dockerfile` is provided so that the user can package the model. To build the image the following command must be issued from the project's root directory:

```
docker build -t parceltour-2-network:latest .
```

## Usage

```
$ tour2network -h
usage: tour2network [-h] [-v] [--flog] [-e ENV] ParcelActivity OUTDIR

tour2network connector

positional arguments:
  ParcelActivity     The path of parcel schedule (tour) output
  OUTDIR             The output directory

optional arguments:
  -h, --help         show this help message and exit
  -v, --verbosity    Increase output verbosity (default: 0)
  --flog             Stores logs to file (default: False)
  -e ENV, --env ENV  Defines the path of the environment file (default: None)
```

## Execution

```
tour2network -vvv --env .env \
    sample-data/input/ParcelSchedule_Test_TotalUrbanDelivery.csv \
    sample-data/output/
```

```
docker run --rm \
  -v $PWD/sample-data:/data \
  --env-file .env \
  parceltour-2-network:latest \
  /data/input/ParcelSchedule_Test_TotalUrbanDelivery.csv \
  /data/output/
```
