"""Processing module
"""

from os.path import join
import os
from os.path import basename
from time import time
from logging import getLogger

import pandas as pd
import numpy as np
import zipfile

logger = getLogger("tour2network.proc")

def run_model(cfg):
    start_time = time()

    logger.info('Importing data...')

    deliveries = pd.read_csv(cfg['ParcelActivity'] )
    deliveries = deliveries[deliveries['VehType'] !='CargoBike' ] # TODO insert as a parameter!
    deliveries.to_csv(join(cfg['OUTDIR'], "ParcelSchedule.csv"), index=False)

    cols = ['ORIG','DEST', 'N_TOT']
    list_files = []
    for tod in range(24):                
        output = deliveries[(deliveries['TripDepTime'] >= tod) & (deliveries['TripDepTime'] < tod+1)].copy()
        output['N_TOT'] = 1
        
        if len(output) > 0:
            pivotTable = pd.pivot_table(output, values=['N_TOT'], index=['O_zone','D_zone'], aggfunc=np.sum)
            pivotTable['ORIG'] = [x[0] for x in pivotTable.index] 
            pivotTable['DEST'] = [x[1] for x in pivotTable.index]
            pivotTable = pivotTable[cols]

            intrazonalTrips = {}
            for i in output[output['N_parcels']>1].index:
                zone = output.at[i,'D_zone']
                if zone in intrazonalTrips.keys():
                    intrazonalTrips[zone] += 1
                else:
                    intrazonalTrips[zone] = 1           
            intrazonalKeys = list(intrazonalTrips.keys())
            for zone in intrazonalKeys:
                if (zone, zone) in pivotTable.index:
                    pivotTable.at[(zone, zone), 'N_TOT'] += intrazonalTrips[zone]
                    del intrazonalTrips[zone]            
            intrazonalTripsDF = pd.DataFrame(np.zeros((len(intrazonalTrips),3)), columns=cols)
            intrazonalTripsDF['ORIG' ] = intrazonalTrips.keys()
            intrazonalTripsDF['DEST' ] = intrazonalTrips.keys()
            intrazonalTripsDF['N_TOT'] = intrazonalTrips.values()
            pivotTable = pivotTable.append(intrazonalTripsDF)
            pivotTable = pivotTable.sort_values(['ORIG','DEST'])
        else:
            pivotTable = pd.DataFrame(columns=cols)
        file_name = join(cfg['OUTDIR'], "tripmatrix_parcels_TOD" + str(tod) +".txt")
        pivotTable.to_csv(file_name, index=False, sep='\t')
        list_files.append(file_name)
    zip_files(cfg, list_files)


def zip_files(cfg, list_files):
    with zipfile.ZipFile(join(cfg['OUTDIR'], 'tripmatrix_parcels_TOD.zip'), 'w') as zipF:
        for file in list_files:
            zipF.write(file, basename(file))
            os.remove(file)
