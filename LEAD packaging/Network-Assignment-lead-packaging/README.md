# Network-Assignment
This module assigns shipments and parcel trips to the road networks

## Introduction

The Network assignemnt module is a static traffic assignment developed to assign trip matrices generated from parcel and shipment scheduling modules to the road networks.the result of this model is the intensities on road networks where the number of truck passing each link can be validated with actual truck counts.

## Installation

The `requirements.txt` and `Pipenv` files are provided for the setup of an environment where the module can be installed. The package includes a `setup.py` file and it can be therefore installed with a `pip install .` when we are at the same working directory as the `setup.py` file. For testing purposes, one can also install the package in editable mode `pip install -e .`.

After the install is completed, an executable will be available to the user.

Furthermore, a `Dockerfile` is provided so that the user can package the model. To build the image the following command must be issued from the project's root directory:

```
docker build -t network-assignment:latest .
```

## Usage

The executable's help message provides information on the parameters that are needed.

```
$ network-assignment -h
usage: network-assignment [-h] [-v] [--flog] [-e ENV]
                          SKIMTIME SKIMDISTANCE NODES ZONES SEGS LINKS SUP_COORDINATES_ID COST_VEHTYPE COST_SOURCING VEHICLE_CAPACITY EMISSIONFACS_BUITENWEG_LEEG
                          EMISSIONFACS_BUITENWEG_VOL EMISSIONFACS_SNELWEG_LEEG EMISSIONFACS_SNELWEG_VOL EMISSIONFACS_STAD_LEEG EMISSIONFACS_STAD_VOL LOGISTIC_SEGMENT VEHICLE_TYPE
                          EMISSION_TYPE TRIPS_VAN_SERVICE TRIPS_VAN_CONSTRUCTION TOURS PARCEL_SCHEDULE TRIP_MATRIX TRIP_MATRIX_PARCELS SHIPMENTS OUTDIR

Network Assignment

positional arguments:
  SKIMTIME              The path of the time skim matrix (mtx)
  SKIMDISTANCE          The path of the distance skim matrix (mtx)
  NODES                 The path of the nodes file (zip)
  ZONES                 The path of the zones shape file (zip)
  SEGS                  The path of the socioeconomics data file (csv)
  LINKS                 The path of the links shape file (zip)
  SUP_COORDINATES_ID    The path of the sup coordinates file (csv)
  COST_VEHTYPE          The path of the Cost_VehType_2016 file (csv)
  COST_SOURCING         The path of the Cost_Sourcing_2016 file (csv)
  VEHICLE_CAPACITY      The path of the CarryingCapacity file (csv)
  EMISSIONFACS_BUITENWEG_LEEG
                        The path of the EmissieFactoren_BUITENWEG_LEEG file (csv)
  EMISSIONFACS_BUITENWEG_VOL
                        The path of the EmissieFactoren_BUITENWEG_VOL file (csv)
  EMISSIONFACS_SNELWEG_LEEG
                        The path of the EmissieFactoren_SNELWEG_LEEG file (csv)
  EMISSIONFACS_SNELWEG_VOL
                        The path of the EmissieFactoren_SNELWEG_VOL file (csv)
  EMISSIONFACS_STAD_LEEG
                        The path of the EmissieFactoren_STAD_LEEG file (csv)
  EMISSIONFACS_STAD_VOL
                        The path of the EmissieFactoren_STAD_VOL file (csv)
  LOGISTIC_SEGMENT      The path of the logistic_segment file (txt)
  VEHICLE_TYPE          The path of the vehicle_type file (txt)
  EMISSION_TYPE         The path of the emission_type file (txt)
  TRIPS_VAN_SERVICE     The path of the TripsVanService file (mtx)
  TRIPS_VAN_CONSTRUCTION
                        The path of the TripsVanConstruction file (mtx)
  TOURS                 The path of the Tours file (csv)
  PARCEL_SCHEDULE       The path of the ParcelSchedule file (csv)
  TRIP_MATRIX           The path of the tripmatrix file (zip)
  TRIP_MATRIX_PARCELS   The path of the tripmatrix_parcels file (zip)
  SHIPMENTS             The path of the Shipments_AfterScheduling file (csv)
  OUTDIR                The output directory

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbosity       Increase output verbosity (default: 0)
  --flog                Stores logs to file (default: False)
  -e ENV, --env ENV     Defines the path of the environment file (default: None)
```

Furthermore, the following parameters must be provided as environment variables either from the environment itself or through a dotenv file that is specified with the `--env <path-to-dotenv>` optional command line argument. An example of the `.env` file and some values is presented below.

```
# string
LABEL="SurfTest"
LABELShipmentTour=""
SELECTED_LINKS=""
N_MULTIROUTE=""
SHIFT_VAN_TO_COMB1=""
IMPEDANCE_SPEED_FREIGHT="V_FR_OS"
IMPEDANCE_SPEED_VAN="V_PA_OS"
# number
N_CPU=2
```

### Examples


In the following examples, it is assumed that the user's terminal is at the project's root directory. Also that all the necessary input files are located in the `sample-data/input` directory and that the `sample-data/output` directory exists.

The user can then execute the model by running the executable.

```
network-assignment -vvv \
    --env .env \
    sample-data/input/skimAfstand_new_REF.mtx \
    sample-data/input/nodes_v5.zip \
    sample-data/input/Zones_v6.zip \
    sample-data/input/links_v5.zip \
    sample-data/input/SupCoordinatesID.csv \
    sample-data/input/Cost_VehType_2016.csv \
    sample-data/input/Cost_Sourcing_2016.csv \
    sample-data/input/CarryingCapacity.csv \
    sample-data/input/EmissieFactoren_BUITENWEG_LEEG.csv \
    sample-data/input/EmissieFactoren_BUITENWEG_VOL.csv \
    sample-data/input/EmissieFactoren_SNELWEG_LEEG.csv \
    sample-data/input/EmissieFactoren_SNELWEG_VOL.csv \
    sample-data/input/EmissieFactoren_STAD_LEEG.csv \
    sample-data/input/EmissieFactoren_STAD_VOL.csv \
    sample-data/input/logistic_segment.txt \
    sample-data/input/vehicle_type.txt \
    sample-data/input/emission_type.txt \
    sample-data/input/TripsVanService.mtx \
    sample-data/input/TripsVanConstruction.mtx \
    sample-data/input/Tours_REF.csv \
    sample-data/input/ParcelSchedule.csv \
    sample-data/input/tripmatrix_REF_TOD.zip \
    sample-data/input/tripmatrix_parcels_REF_TOD.zip \
    sample-data/input/Shipments_AfterScheduling_REF.csv \
    sample-data/output/
```

If the package installation has been omitted, the model can of course also be run with `python -m src.networkassignment.__main__ <args>`.

Finally, the model can be executed with `docker run`:

```
docker run --rm \
  -v $PWD/sample-data:/data \
  --env-file .env \
  registry.gitlab.com/inlecom/lead/models/network-assignment:latest \
  /data/input/skimAfstand_new_REF.mtx \
  /data/input/nodes_v5.zip \
  /data/input/Zones_v6.zip \
  /data/input/links_v5.zip \
  /data/input/SupCoordinatesID.csv \
  /data/input/Cost_VehType_2016.csv \
  /data/input/Cost_Sourcing_2016.csv \
  /data/input/CarryingCapacity.csv \
  /data/input/EmissieFactoren_BUITENWEG_LEEG.csv \
  /data/input/EmissieFactoren_BUITENWEG_VOL.csv \
  /data/input/EmissieFactoren_SNELWEG_LEEG.csv \
  /data/input/EmissieFactoren_SNELWEG_VOL.csv \
  /data/input/EmissieFactoren_STAD_LEEG.csv \
  /data/input/EmissieFactoren_STAD_VOL.csv \
  /data/input/logistic_segment.txt \
  /data/input/vehicle_type.txt \
  /data/input/emission_type.txt \
  /data/input/TripsVanService.mtx \
  /data/input/TripsVanConstruction.mtx \
  /data/input/Tours_REF.csv \
  /data/input/ParcelSchedule.csv \
  /data/input/tripmatrix_REF_TOD.zip \
  /data/input/tripmatrix_parcels_REF_TOD.zip \
  /data/input/Shipments_AfterScheduling_REF.csv \
  /data/output/
```
