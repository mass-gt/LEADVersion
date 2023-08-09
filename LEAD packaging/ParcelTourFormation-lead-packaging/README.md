# Parcel Tour Formation

_Parcel Tour Formation model for the Living Lab of The Hague for the LEAD platform._

## Installation

The `requirements.txt` and `Pipenv` files are provided for the setup of an environment where the module can be installed. The package includes a `setup.py` file and it can be therefore installed with a `pip install .` when we are at the same working directory as the `setup.py` file. For testing purposes, one can also install the package in editable mode `pip install -e .`.

After the install is completed, an executable will be available to the user.

Furthermore, a `Dockerfile` is provided so that the user can package the model. To build the image the following command must be issued from the project's root directory:

```
docker build -t parcel-tour-formation:latest .
```

## Usage

The executable's help message provides information on the parameters that are needed.

```
$ parcel-tour-formation -h
usage: parcel-tour-formation [-h] [-v] [--flog] [-e ENV] PARCELS PARCELS_HUB2HUB SKIMTIME SKIMDISTANCE ZONES SEGS PARCELNODES DEPARTURE_TIME_PARCELS_CDF SUP_COORDINATES OUTDIR

Parcel Tour Formation

Some more info.

positional arguments:
  PARCELS               The path of the demand parcel fulfilment file (csv)
  PARCELS_HUB2HUB       The path of the demand parcel fulfilment file (csv)
  SKIMTIME              The path of the time skim matrix (mtx)
  SKIMDISTANCE          The path of the distance skim matrix (mtx)
  ZONES                 The path of the area shape file (shp)
  SEGS                  The path of the socioeconomics data file (csv)
  PARCELNODES           The path of the parcel nodes file (shp)
  DEPARTURE_TIME_PARCELS_CDF
                        The path of the departure time parcels CDF file (csv)
  SUP_COORDINATES       The path of the SUP coordinates file (csv)
  OUTDIR                The output directory

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbosity       Increase output verbosity (default: 0)
  --flog                Stores logs to file (default: False)
  -e ENV, --env ENV     Defines the path of the environment file (default: None)
```

Furthermore, the following parameters must be provided as environment variables either from the environment itself or through a dotenv file that is specified with the `--env <path-to-dotenv>` optional command line argument. An example of the `.env` file and some values is presented below.

```
# string parameters
LABEL=test

# boolean parameters
COMBINE_DELIVERY_PICKUP_TOUR=True
CROWDSHIPPING=False
CONSOLIDATED_TRIPS=True

# numeric parameters
PARCELS_MAXLOAD=180
PARCELS_DROPTIME=120
PARCELS_MAXLOAD_Hub2Hub=500
PARCELS_DROPTIME_Hub2Hub=90

# string list parameters
Gemeenten_studyarea=sGravenhage
Gemeenten_CS=sGravenhage,Zoetermeer,Midden_Delfland
```

### Examples

In the following examples, it is assumed that the user's terminal is at the project's root directory. Also that all the necessary input files are located in the `sample-data/input` directory and that the `sample-data/output` directory exists.

The user can then execute the model by running the executable.

```
parcel-tour-formation -vvv --env .env \
    sample-data/input/ParcelDemand.csv \
    sample-data/input/ParcelDemand_Hub2Hub.csv \
    sample-data/input/skimTijd_new_REF.mtx \
    sample-data/input/skimAfstand_new_REF.mtx \
    sample-data/input/Zones_v4.zip \
    sample-data/input/SEGS2020.csv \
    sample-data/input/parcelNodes_v2Cycloon.zip \
    sample-data/input/departureTimeParcelsCDF.csv \
    sample-data/input/SupCoordinatesID.csv \
    sample-data/output/
```

If the package installation has been omitted, the model can of course also be run with `python -m src.parceltourformation.__main__ <args>`.

Finally, the model can be executed with `docker run`:

```
docker run --rm \
  -v $PWD/sample-data:/data \
  --env-file .env \
  parcel-tour-formation:latest \
  /data/input/ParcelDemand.csv \
  /data/input/ParcelDemand_Hub2Hub.csv \
  /data/input/skimTijd_new_REF.mtx \
  /data/input/skimAfstand_new_REF.mtx \
  /data/input/Zones_v4.zip \
  /data/input/SEGS2020.csv \
  /data/input/parcelNodes_v2Cycloon.zip \
  /data/input/departureTimeParcelsCDF.csv \
  /data/input/SupCoordinatesID.csv \
  /data/output
```