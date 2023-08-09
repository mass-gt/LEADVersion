# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 10:08:06 2022

@author: rtapia
"""



from __functions__ import read_mtx, read_shape, create_geojson, get_traveltime, get_distance
import pandas as pd
import numpy as np
# import networkx as nx
# from itertools import islice, tee
# import math
import sys, os
# import time
import ast
import datetime as dt
import json






class HiddenPrints: #
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
#%%


varDict = {}
'''FOR ALL MODULES'''
cwd = os.getcwd().replace(os.sep, '/')
datapath = cwd.replace('Code', '')# + "/Parcel_Market"



#%% Define all variables
def generate_args(method):
    varDict = {}
   
    '''FOR ALL MODULES'''
    cwd = os.getcwd().replace(os.sep, '/')
    datapath = cwd.replace('Code', '')
    
    if method == 'from_file':   
            
        if sys.argv[0] == '':
            params_file = open(f'{datapath}/Input/Params_Conn_Sched2Ntw.txt')
            
            # This are the defaults, might need to change for console run!!!
            varDict['LABEL'	]			= 'FrancescoTest'				
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ 'Input' +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ 'Output' +'/'			
            
            varDict['ParcelActivity']              = varDict['INPUTFOLDER'] + "ParcelSchedule_Test_TotalUrbanDelivery.csv"    

            
        else:  # This is the part for line cod execution
            locationparam = f'{datapath}'+'/' + sys.argv[2] +'/' + sys.argv[4]
            params_file = open(locationparam)
            varDict['LABEL'	]			= sys.argv[1]				
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[2] +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[3] +'/'			
            
            varDict['ParcelActivity']              = varDict['INPUTFOLDER'] + sys.argv[5]     

            pd.options.mode.chained_assignment = None # So, it shuts up the warnings (remove when running in spyder)
       

        for line in params_file:
            if len(line.split('=')) > 1:
                key, value = line.split('=')
                if len(value.split(';')) > 1:
                    # print(value)
                    value, dtype = value.split(';')
                    if len(dtype.split('#')) > 1: dtype, comment = dtype.split('#')
                    # Allow for spacebars around keys, values and dtypes
                    while key[0] == ' ' or key[0] == '\t': key = key[1:]
                    while key[-1] == ' ' or key[-1] == '\t': key = key[0:-1]
                    while value[0] == ' ' or value[0] == '\t': value = value[1:]
                    while value[-1] == ' ' or value[-1] == '\t': value = value[0:-1]
                    while dtype[0] == ' ' or dtype[0] == '\t': dtype = dtype[1:]
                    while dtype[-1] == ' ' or dtype[-1] == '\t': dtype = dtype[0:-1]
                    dtype = dtype.replace('\n',"")
                    # print(key, value, dtype)
                    if dtype == 'string': varDict[key] = str(value)
                    elif dtype == 'list': varDict[key] = ast.literal_eval(value)
                    elif dtype == 'int': varDict[key] = int(value)               
                    elif dtype == 'float': varDict[key] = float(value)               
                    elif dtype == 'bool': varDict[key] = eval(value)               
                    elif dtype == 'variable': varDict[key] = globals()[value]
                    elif dtype == 'eval': varDict[key] = eval(value)


   
    
    args = ['', varDict]
    return args, varDict
Comienzo = dt.datetime.now()
print ("Comienzo: ",Comienzo)    

method = 'from_file' #either from_file or from_code
args, varDict = generate_args(method)


#%% Import data




deliveries = pd.read_csv(varDict['ParcelActivity'] )


#%% Filter cargobikes if wanted

deliveries = deliveries[deliveries['VehType'] !='CargoBike' ] # TODO insert as a parameter!


deliveries.to_csv(f"{varDict['OUTPUTFOLDER']}ParcelSchedule_{varDict['LABEL']}.csv", index=False)


#%% Generate TOD Matrix


np.random.seed(int(varDict['Seed']))
cols = ['ORIG','DEST', 'N_TOT']

for tod in range(24):                
    # print(f'\t Also generating trip matrix for TOD {tod}...'), log_file.write(f'\t Also generating trip matrix for TOD {tod}...\n')
    output = deliveries[(deliveries['TripDepTime'] >= tod) & (deliveries['TripDepTime'] < tod+1)].copy()
    output['N_TOT'] = 1
    
    if len(output) > 0:
        # Gebruik deze dummies om het aantal ritten per HB te bepalen, voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(output, values=['N_TOT'], index=['O_zone','D_zone'], aggfunc=np.sum)
        pivotTable['ORIG'] = [x[0] for x in pivotTable.index] 
        pivotTable['DEST'] = [x[1] for x in pivotTable.index]
        pivotTable = pivotTable[cols]

        # Assume one intrazonal trip for each zone with multiple deliveries visited in a tour
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
        
    pivotTable.to_csv(f"{varDict['OUTPUTFOLDER']}tripmatrix_parcels_{varDict['LABEL']}_TOD{tod}.txt", index=False, sep='\t')

#%% Export to input excel files of copert


















