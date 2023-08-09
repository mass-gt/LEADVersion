# Connector gen-2-market

## Installation

The `requirements.txt` and `Pipenv` files are provided for the setup of an environment where the module can be installed. The package includes a `setup.py` file and it can be therefore installed with a `pip install .` when we are at the same working directory as the `setup.py` file. For testing purposes, one can also install the package in editable mode `pip install -e .`.

After the install is completed, an executable will be available to the user.

Furthermore, a `Dockerfile` is provided so that the user can package the model. To build the image the following command must be issued from the project's root directory:

```
docker build -t gen2mkt:latest .
```

## Usage

```
$ gen2mkt -h
usage: gen2mkt [-h] [-v] [--flog] [-e ENV] PARCELS ZONES SEGS OUTDIR

Parcel Generation to Parcel Market connector.

positional arguments:
  PARCELS            The path of the parcel demand file (csv)
  ZONES              The path of the area shape file (shp)
  SEGS               The path of the socioeconomics data file (csv)
  OUTDIR             The output directory

optional arguments:
  -h, --help         show this help message and exit
  -v, --verbosity    Increase output verbosity (default: 0)
  --flog             Stores logs to file (default: False)
  -e ENV, --env ENV  Defines the path of the environment file (default: None)
```

Furthermore, the following parameters must be provided as environment variables either from the environment itself or through a dotenv file that is specified with the `--env <path-to-dotenv>` optional command line argument. An example of the `.env` file and some values is presented below.

```
# numeric parameters
Local2Local=0.04
CS_cust_willingness=0.1
PARCELS_PER_HH_B2C=0.0078000000000000005
PARCELS_M=20.8
PARCELS_DAYS=250
PARCELS_M_HHS=8.0

# string list parameters
Gemeenten_studyarea=sGravenhage
```

## Execution

```
gen2mkt -vvv --env .env \
    sample-data/input/ParcelDemand.csv \
    sample-data/input/Zones_v4.zip \
    sample-data/input/SEGS2020.csv \
    sample-data/output/
```

```
docker run --rm \
  -v $PWD/sample-data:/data \
  --env-file .env \
  gen2mkt:latest \
  /data/input/ParcelDemand.csv \
  /data/input/Zones_v4.zip \
  /data/input/SEGS2020.csv \
  /data/output/
```
