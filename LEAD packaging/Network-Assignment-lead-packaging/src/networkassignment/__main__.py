"""
Network Assignment
"""

import logging
from logging.handlers import RotatingFileHandler
from os import environ
from os.path import isfile, join, abspath, exists
from sys import argv
from argparse import (ArgumentParser, RawTextHelpFormatter,
                      ArgumentDefaultsHelpFormatter, ArgumentTypeError)

from dotenv import dotenv_values

from .envctl import parse_env_values
from .proc import run_model


LOG_FILE_MAX_BYTES = 50e6
LOG_MSG_FMT = "%(asctime)s %(levelname)-8s %(name)s \
%(filename)s#L%(lineno)d %(message)s"
LOG_DT_FMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("networkassignment")


class RawDefaultsHelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """Argparse formatter class"""


def strfile(path):
    """Argparse type checking method
    string path for file should exist"""
    if isfile(path):
        return path
    raise ArgumentTypeError("Input file does not exist")


def strdir(path):
    """Argparse type checking method
    string path for file should exist"""
    if exists(path):
        return path
    raise ArgumentTypeError("Input directory does not exist")


def get_log_level(vcount):
    """Translates the CLI input of the user for the verbosity
    to an actual logging level.

    :param vcount: The user input in verbosity counts
    :type vcount: int
    :return: The logging level constant
    :rtype: int
    """
    loglevel = logging.ERROR
    if vcount >= 3:
        loglevel = logging.DEBUG
    elif vcount == 2:
        loglevel = logging.INFO
    elif vcount == 1:
        loglevel = logging.WARNING
    else:
        return loglevel

    return loglevel


def main():
    """Main method of Network Assignment Model.
    """

    # command line argument parsing
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDefaultsHelpFormatter)

    parser.add_argument('SKIMDISTANCE', type=strfile,
                        help='The path of the distance skim matrix (mtx)')
    parser.add_argument('NODES', type=strfile,
                        help='The path of the nodes file (zip)')
    parser.add_argument('ZONES', type=strfile,
                        help='The path of the zones shape file (zip)')
    parser.add_argument('LINKS', type=strfile,
                        help='The path of the links shape file (zip)')
    parser.add_argument('SUP_COORDINATES_ID', type=strfile,
                        help='The path of the sup coordinates file (csv)')
    parser.add_argument('COST_VEHTYPE', type=strfile,
                        help='The path of the Cost_VehType_2016 file (csv)')
    parser.add_argument('COST_SOURCING', type=strfile,
                        help='The path of the Cost_Sourcing_2016 file (csv)')
    parser.add_argument('VEHICLE_CAPACITY', type=strfile,
                        help='The path of the CarryingCapacity file (csv)')
    parser.add_argument('EMISSIONFACS_BUITENWEG_LEEG', type=strfile,
                        help='The path of the EmissieFactoren_BUITENWEG_LEEG file (csv)')
    parser.add_argument('EMISSIONFACS_BUITENWEG_VOL', type=strfile,
                        help='The path of the EmissieFactoren_BUITENWEG_VOL file (csv)')
    parser.add_argument('EMISSIONFACS_SNELWEG_LEEG', type=strfile,
                        help='The path of the EmissieFactoren_SNELWEG_LEEG file (csv)')
    parser.add_argument('EMISSIONFACS_SNELWEG_VOL', type=strfile,
                        help='The path of the EmissieFactoren_SNELWEG_VOL file (csv)')
    parser.add_argument('EMISSIONFACS_STAD_LEEG', type=strfile,
                        help='The path of the EmissieFactoren_STAD_LEEG file (csv)')
    parser.add_argument('EMISSIONFACS_STAD_VOL', type=strfile,
                        help='The path of the EmissieFactoren_STAD_VOL file (csv)')
    parser.add_argument('LOGISTIC_SEGMENT', type=strfile,
                        help='The path of the logistic_segment file (txt)')
    parser.add_argument('VEHICLE_TYPE', type=strfile,
                        help='The path of the vehicle_type file (txt)')
    parser.add_argument('EMISSION_TYPE', type=strfile,
                        help='The path of the emission_type file (txt)')
    parser.add_argument('TRIPS_VAN_SERVICE', type=strfile,
                        help='The path of the TripsVanService file (mtx)')
    parser.add_argument('TRIPS_VAN_CONSTRUCTION', type=strfile,
                        help='The path of the TripsVanConstruction file (mtx)')
    parser.add_argument('TOURS', type=strfile,
                        help='The path of the Tours file (csv)')
    parser.add_argument('PARCEL_SCHEDULE', type=strfile,
                        help='The path of the ParcelSchedule file (csv)')
    parser.add_argument('TRIP_MATRIX', type=strfile,
                        help='The path of the tripmatrix file (zip)')
    parser.add_argument('TRIP_MATRIX_PARCELS', type=strfile,
                        help='The path of the tripmatrix_parcels file (zip)')
    parser.add_argument('SHIPMENTS', type=strfile,
                        help='The path of the Shipments_AfterScheduling file (csv)')
    parser.add_argument('OUTDIR', type=strdir, help='The output directory')

    parser.add_argument('-v', '--verbosity', action='count', default=0,
                        help='Increase output verbosity')
    parser.add_argument('--flog', action='store_true', default=False,
                        help='Stores logs to file')
    parser.add_argument('-e', '--env', type=str, default=None,
                        help='Defines the path of the environment file')

    args = parser.parse_args(argv[1:])

    # setting of the logger
    formatter = logging.Formatter(fmt=LOG_MSG_FMT, datefmt=LOG_DT_FMT)
    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    logger.addHandler(shandler)
    if args.flog:
        fhandler = RotatingFileHandler(
            join(args.OUTDIR, "logs.txt"),
            mode='w',
            backupCount=1,
            maxBytes=LOG_FILE_MAX_BYTES
        )
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)

    loglevel = get_log_level(args.verbosity)
    logger.setLevel(loglevel)

    logger.debug('CMD : %s', ' '.join(argv))
    logger.debug('ARGS: %s', args)

    # setting of the configuration
    config = vars(args).copy()
    _ = [config.pop(key) for key in ("verbosity", "flog", "env")]
    config_env = {}
    if args.env:
        if isfile(abspath(args.env)):
            logger.info("using env file: %s", abspath(args.env))
            config_env = parse_env_values(dotenv_values(abspath(args.env)))
        else:
            raise ValueError('error: invalid .env file')
    else:
        logger.info("using environment")
        config_env = parse_env_values(environ)
    config.update(config_env)
    logger.debug('CONFIG: %s', config)

    for key, value in config.items():
        print(f'{key:<30s}: {value}')

    run_model(config)


if __name__ == "__main__":
    main()
