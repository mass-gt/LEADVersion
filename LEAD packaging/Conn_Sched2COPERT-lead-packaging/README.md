# ParcelTourFormation-2-COPERT

Receives the output from the Parcel Tour Formation model, and with additional weather data transforms it in a format compatible with the input of the COPERT model.

## Introduction


## Installation

The `requirements.txt` and `Pipenv` files are provided for the setup of an environment where the module can be installed. The package includes a `setup.py` file and it can be therefore installed with a `pip install .` when we are at the same working directory as the `setup.py` file. For testing purposes, one can also install the package in editable mode `pip install -e .`.

After the install is completed, an executable will be available to the user.

Furthermore, a `Dockerfile` is provided so that the user can package the model. To build the image the following command must be issued from the project's root directory:

```
docker build -t parceltour-2-copert:latest .
```

## Usage

The executable's help message provides information on the parameters that are needed.

```
$ tour-2-copert -h
usage: tour-2-copert [-h] [-v] [--flog] [-e ENV] parcel_activity weather outdir

Parcel Tour Formation To COPERT connector.

Receives the output from the Parcel Tour Formation model, and with additional weather data
transforms it in a format compatible with the input of the COPERT model.

positional arguments:
  parcel_activity    The output of the Parcel Tour Formation model (csv)
  weather            The path of weather file (csv)
  outdir             The output directory

optional arguments:
  -h, --help         show this help message and exit
  -v, --verbosity    Increase output verbosity (default: 0)
  --flog             Stores logs to file (default: False)
  -e ENV, --env ENV  Defines the path of the environment file (default: None)
```

Furthermore, the following parameters must be provided as environment variables either from the environment itself or through a dotenv file that is specified with the `--env <path-to-dotenv>` optional command line argument. An example of the `.env` file and some values is presented below.

```
# boolean
ElectricEmissions=True
# number
Year=2021
PeakHourMorningStart=7
PeakHourMorningFinish=10
PeakHourAfternoonStart=16
PeakHourAfternoonFinish=19
# JSON
VehicleType={"Van": {"Van1": 0.8, "Hybrid": 0.2} , "ConsoldVan": {"Van2": 0.8, "Electric": 0.1, "Hybrid": 0.1}, "CargoBike": {"Bike": 1}}
VehicleEuroStand={"Van1": "Euro 6 d", "Van2": "Euro 4", "Electric": 300, "Hybrid": ["Euro 6 d", 300, 0.8]}
VehicleCat={"Van1": "Light Commercial Vehicles", "Van2": "Light Commercial Vehicles", "Electric": 0,"Hybrid": "Passenger Cars", "Bike": 0}
VehicleFuel={"Van1": "Diesel", "Van2": "Diesel", "Electric": 0, "Hybrid": "Petrol Hybrid", "Bike": 0}
VehicleSegment={"Van1": "N1-III", "Van2": "N1-III", "Electric": 0, "Hybrid": "Large-SUV-Executive", "Bike": 0}
VehicleOffPeakSpeed={"Van1": 60 , "Van2": 60, "Electric": 60 ,"Hybrid": 60, "Bike": 20}
VehiclePeakSpeed={"Van1": 30, "Van2": 30, "Electric": 20, "Hybrid": 30, "Bike": 20}
```

### Examples

In the following examples, it is assumed that the user's terminal is at the project's root directory. Also that all the necessary input files are located in the `sample-data/input` directory and that the `sample-data/output` directory exists.

The user can then execute the model by running the executable.

```
tour-2-copert -vvv \
    --env .env \
    sample-data/input/ParcelSchedule.csv \
    sample-data/input/Weather.csv \
    sample-data/output/
```

If the package installation has been omitted, the model can of course also be run with `python -m src.tour2copert.__main__ <args>`.

Finally, the model can be executed with `docker run`:

```
docker run --rm \
  -v $PWD/sample-data:/data \
  --env-file .env \
  parceltour-2-copert:latest \
  /data/input/ParcelSchedule.csv \
  /data/input/Weather.csv \
  /data/output/
```
