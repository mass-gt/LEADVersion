# Shipment_Synthesis

## Introduction

This module Synthesize the shipments between origins and destinations of the commodities.

## Installation


## Installation

The `requirements.txt` and `Pipenv` files are provided for the setup of an environment where the module can be installed. The package includes a `setup.py` file and it can be therefore installed with a `pip install .` when we are at the same working directory as the `setup.py` file. For testing purposes, one can also install the package in editable mode `pip install -e .`.

After the install is completed, an executable `shipment-synthesis` will be available to the user.

Furthermore, a `Dockerfile` is provided so that the user can package the model. To build the image the following command must be issued from the project's root directory:

```
docker build -t shipment-synthesis:latest .
```

## Usage

The executable's help message provides information on the parameters that are needed.

```
$ shipment-synthesis -h
usage: shipment-synthesis [-h] [-v] [--flog] [-e ENV]
                          SKIMTIME SKIMDISTANCE NODES ZONES SEGS PARCELNODES DISTRIBUTIECENTRA NSTR_TO_LS MAKE_DISTRIBUTION USE_DISTRIBUTION SUP_COORDINATES_ID CORRECTIONS_TONNES
                          CEP_SHARES COST_VEHTYPE COST_SOURCING NUTS3_TO_MRDH VEHICLE_CAPACITY LOGISTIC_FLOWTYPES PARAMS_TOD PARAMS_SSVT PARAMS_ET_FIRST PARAMS_ET_LATER
                          ZEZ_CONSOLIDATION ZEZ_SCENARIO FIRMS_REF NSTR LOGSEG SHIP_SIZE VEH_TYPE FLOW_TYPE COMMODITY_MTX OUTDIR

Shipment Synthesis

positional arguments:
  SKIMTIME            The path of the time skim matrix (mtx)
  SKIMDISTANCE        The path of the distance skim matrix (mtx)
  NODES               The path of the logistics nodes shape file (shp)
  ZONES               The path of the study area shape file (shp)
  SEGS                The path of the socioeconomics data file (csv)
  PARCELNODES         The path of the parcel depot nodes file (shp)
  DISTRIBUTIECENTRA   The path of the distribution centres file (csv)
  NSTR_TO_LS          The path of the conversion NSTR to Logistics segments file (csv)
  MAKE_DISTRIBUTION   The path of the Making Shipments per logistic sector file (csv)
  USE_DISTRIBUTION    The path of the Using Shipments per logistic sector file (csv)
  SUP_COORDINATES_ID  The path of the SUP coordinates file (csv)
  CORRECTIONS_TONNES  The path of the Correction of Tonnes file (csv)
  CEP_SHARES          The path of the courier market shares file (csv)
  COST_VEHTYPE        The path of the costs per vehicles types file (csv)
  COST_SOURCING       The path of the costs per vehicles types file (csv)
  NUTS3_TO_MRDH       The path of the conversion NUTS to MRDH file (csv)
  VEHICLE_CAPACITY    The path of the carrying capacity file (csv)
  LOGISTIC_FLOWTYPES  The path of the markete share of logistic flow types file (csv)
  PARAMS_TOD          The path of the Time Of Day choice model parameters file (csv)
  PARAMS_SSVT         The path of the shipment size and vehicle type choice model parameters file (csv)
  PARAMS_ET_FIRST     The path of the End pf Tour model parameters file for the first visited location (csv)
  PARAMS_ET_LATER     The path of the End pf Tour model parameters file for the later visited location (csv)
  ZEZ_CONSOLIDATION   The path of the Consolidation Potentials for different logistics sectors file (csv)
  ZEZ_SCENARIO        The path of the specifications for zero emission zones in the study area file (csv)
  FIRMS_REF           The path of the specifications of synthesized firms file (csv)
  NSTR                (txt)
  LOGSEG              (txt)
  SHIP_SIZE           (txt)
  VEH_TYPE            (txt)
  FLOW_TYPE           (txt)
  COMMODITY_MTX       (csv)
  OUTDIR              The output directory

optional arguments:
  -h, --help          show this help message and exit
  -v, --verbosity     Increase output verbosity (default: 0)
  --flog              Stores logs to file (default: False)
  -e ENV, --env ENV   Defines the path of the environment file (default: None)
```

Furthermore, the following parameters must be provided as environment variables either from the environment itself or through a dotenv file that is specified with the `--env <path-to-dotenv>` command line argument. An example of the `.env` file and some values is presented below.
```
# string parameters
LABEL="REF"
SHIPMENTS_REF=""
FAC_LS0=""
FAC_LS1=""
FAC_LS2=""
FAC_LS3=""
FAC_LS4=""
FAC_LS5=""
FAC_LS6=""
FAC_LS7=""

# boolean parameters
NEAREST_DC=""

# numeric parameters
YEARFACTOR=209
PARCELS_GROWTHFREIGHT=1.0

# string list parameters

# numeric list parameters

# json
```

## Examples

In the following examples, it is assumed that the user's terminal is at the project's root directory. Also that all the necessary input files are located in the `sample-data/inputs` directory and that the `sample-data/outputs` directory exists.

The user can then execute the model by running the executable.

```
shipment-synthesis -vvv --env .env \
    sample-data/input/skimTijd_new_REF.mtx \
    sample-data/input/skimAfstand_new_REF.mtx \
    sample-data/input/nodes_v5.shp \
    sample-data/input/Zones_v6.shp \
    sample-data/input/SEGS2020.csv \
    sample-data/input/parcelNodes_v2.shp \
    sample-data/input/distributieCentra.csv \
    sample-data/input/nstrToLogisticSegment.csv \
    sample-data/input/MakeDistribution.csv \
    sample-data/input/UseDistribution.csv \
    sample-data/input/SupCoordinatesID.csv \
    sample-data/input/CorrectionsTonnes2016.csv \
    sample-data/input/CEPshares.csv \
    sample-data/input/Cost_VehType_2016.csv \
    sample-data/input/Cost_Sourcing_2016.csv \
    sample-data/input/NUTS32013toMRDH.csv \
    sample-data/input/CarryingCapacity.csv \
    sample-data/input/LogFlowtype_Shares.csv \
    sample-data/input/Params_TOD.csv \
    sample-data/input/Params_ShipSize_VehType.csv \
    sample-data/input/Params_EndTourFirst.csv \
    sample-data/input/Params_EndTourLater.csv \
    sample-data/input/ConsolidationPotential.csv \
    sample-data/input/ZEZscenario.csv \
    sample-data/input/Firms.csv \
    sample-data/input/nstr.txt \
    sample-data/input/logistic_segment.txt \
    sample-data/input/shipment_size.txt \
    sample-data/input/vehicle_type.txt \
    sample-data/input/flow_type.txt \
    sample-data/input/CommodityMatrixNUTS3.csv \
    sample-data/output
```

If the package installation has been omitted, the model can of course also be run with `python -m src.shipmentsynth.__main__ <args>`.

Finally, the model can be executed with `docker run`:

```
docker run --rm \
  -v $PWD/sample-data/input:/data/input \
  -v $PWD/sample-data/output:/data/output \
  --env-file .env \
  shipment-synthesis:latest \
  /data/input/skimTijd_new_REF.mtx \
  /data/input/skimAfstand_new_REF.mtx \
  /data/input/nodes_v5.shp \
  /data/input/Zones_v6.shp \
  /data/input/SEGS2020.csv \
  /data/input/parcelNodes_v2.shp \
  /data/input/distributieCentra.csv \
  /data/input/nstrToLogisticSegment.csv \
  /data/input/MakeDistribution.csv \
  /data/input/UseDistribution.csv \
  /data/input/SupCoordinatesID.csv \
  /data/input/CorrectionsTonnes2016.csv \
  /data/input/CEPshares.csv \
  /data/input/Cost_VehType_2016.csv \
  /data/input/Cost_Sourcing_2016.csv \
  /data/input/NUTS32013toMRDH.csv \
  /data/input/CarryingCapacity.csv \
  /data/input/LogFlowtype_Shares.csv \
  /data/input/Params_TOD.csv \
  /data/input/Params_ShipSize_VehType.csv \
  /data/input/Params_EndTourFirst.csv \
  /data/input/Params_EndTourLater.csv \
  /data/input/ConsolidationPotential.csv \
  /data/input/ZEZscenario.csv \
  /data/input/Firms.csv \
  /data/input/nstr.txt \
  /data/input/logistic_segment.txt \
  /data/input/shipment_size.txt \
  /data/input/vehicle_type.txt \
  /data/input/flow_type.txt \
  /data/input/CommodityMatrixNUTS3.csv \
  /data/output
```

Temporary example:

```
python3 __module_SHIP__.py REF Input Output \
    skimTijd_new_REF.mtx \
    skimAfstand_new_REF.mtx \
    nodes_v5.shp \
    Zones_v6.shp \
    SEGS2020.csv \
    parcelNodes_v2.shp \
    distributieCentra.csv \
    nstrToLogisticSegment.csv \
    MakeDistribution.csv \
    UseDistribution.csv \
    SupCoordinatesID.csv \
    CorrectionsTonnes2016.csv \
    CEPshares.csv \
    Cost_VehType_2016.csv \
    Cost_Sourcing_2016.csv \
    NUTS32013toMRDH.csv \
    CarryingCapacity.csv \
    LogFlowtype_Shares.csv \
    Params_TOD.csv \
    Params_ShipSize_VehType.csv \
    Params_EndTourFirst.csv \
    Params_EndTourLater.csv \
    ConsolidationPotential.csv \
    ZEZscenario.csv \
    Firms.csv
```

