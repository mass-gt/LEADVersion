"""
Shipment Synthesis
"""

import logging
from logging.handlers import RotatingFileHandler
from os import environ
from os.path import isfile, join, exists, abspath
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

logger = logging.getLogger("shipmentsynth")


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
    """Main method of Shipment Synthesis Model.
    """

    # command line argument parsing
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDefaultsHelpFormatter)

    parser.add_argument('SKIMTIME', type=strfile, help='The path of the time skim matrix (mtx)')
    parser.add_argument('SKIMDISTANCE', type=strfile,
                        help='The path of the distance skim matrix (mtx)')
    parser.add_argument('NODES', type=strfile, help='The path of the logistics nodes shape file (shp)')
    parser.add_argument('ZONES', type=strfile, help='The path of the study area shape file (shp)')
    parser.add_argument('SEGS', type=strfile, help='The path of the socioeconomics data file (csv)')
    parser.add_argument('PARCELNODES', type=strfile,
                        help='The path of the parcel depot nodes file (shp)')
    parser.add_argument('DISTRIBUTIECENTRA', type=strfile,
                        help='The path of the distribution centres file (csv)')
    parser.add_argument('NSTR_TO_LS', type=strfile,
                        help='The path of the conversion NSTR to Logistics segments file (csv)')
    parser.add_argument('MAKE_DISTRIBUTION', type=strfile,
                        help='The path of the Making Shipments per logistic sector file (csv)')
    parser.add_argument('USE_DISTRIBUTION', type=strfile,
                        help='The path of the Using Shipments per logistic sector file (csv)')
    parser.add_argument('SUP_COORDINATES_ID', type=strfile,
                        help='The path of the SUP coordinates file (csv)')
    parser.add_argument('CORRECTIONS_TONNES', type=strfile,
                        help='The path of the Correction of Tonnes file (csv)')
    parser.add_argument('CEP_SHARES', type=strfile,
                        help='The path of the courier market shares file (csv)')
    parser.add_argument('COST_VEHTYPE', type=strfile,
                        help='The path of the costs per vehicles types file (csv)')
    parser.add_argument('COST_SOURCING', type=strfile,
                        help='The path of the costs per vehicles types file (csv)')
    parser.add_argument('NUTS3_TO_MRDH', type=strfile,
                        help='The path of the conversion NUTS to MRDH file (csv)')
    parser.add_argument('VEHICLE_CAPACITY', type=strfile,
                        help='The path of the carrying capacity file (csv)')
    parser.add_argument('LOGISTIC_FLOWTYPES', type=strfile,
                        help='The path of the markete share of logistic flow types file (csv)')
    parser.add_argument('PARAMS_TOD', type=strfile,
                        help='The path of the Time Of Day choice model parameters file (csv)')
    parser.add_argument('PARAMS_SSVT', type=strfile,
                        help='The path of the shipment size and vehicle type '
                             'choice model parameters file (csv)')
    parser.add_argument('PARAMS_ET_FIRST', type=strfile,
                        help='The path of the End pf Tour model parameters file '
                             'for the first visited location (csv)')
    parser.add_argument('PARAMS_ET_LATER', type=strfile,
                        help='The path of the End pf Tour model parameters file '
                             'for the later visited location (csv)')
    parser.add_argument('ZEZ_CONSOLIDATION', type=strfile,
                        help='The path of the Consolidation Potentials for different logistics '
                             'sectors file (csv)')
    parser.add_argument('ZEZ_SCENARIO', type=strfile,
                        help='The path of the specifications for zero emission zones in the study '
                             'area file (csv)')
    parser.add_argument('FIRMS_REF', type=strfile,
                        help='The path of the specifications of synthesized firms file (csv)')

    parser.add_argument('NSTR', type=strfile,
                        help='(txt)')
    parser.add_argument('LOGSEG', type=strfile,
                        help='(txt)')
    parser.add_argument('SHIP_SIZE', type=strfile,
                        help='(txt)')
    parser.add_argument('VEH_TYPE', type=strfile,
                        help='(txt)')
    parser.add_argument('FLOW_TYPE', type=strfile,
                        help='(txt)')
    parser.add_argument('COMMODITY_MTX', type=strfile,
                        help='(csv)')

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
