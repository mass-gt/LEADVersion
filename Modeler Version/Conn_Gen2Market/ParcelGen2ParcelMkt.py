# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:00:14 2021

@author: rtapia
"""
from __functions__ import read_mtx, read_shape, create_geojson, get_traveltime, get_distance
import pandas as pd
import numpy as np
import networkx as nx
from itertools import islice, tee
import math
import sys, os
import time
import ast
import datetime as dt

# from  StartUp import *


# sys.argv = ['ParcelGen2ParcelMkt',
#             'test',
#             'input',
#             'output',
#             'Params_ParcelGen.txt' ,
#             'ParcelDemand_Test.csv' ,
#             'skimTijd_new_REF.mtx' ,
#             'skimAfstand_new_REF.mtx' ,
#             'Zones_v4.shp' ,
#            'SEGS2020.csv',
#             'parcelNodes_v2.shp'
    
#     ]

#%% Here I use the pure generation of 
cwd = os.getcwd().replace(os.sep, '/')
datapath = cwd.replace('Code', '')        
varDict = {}
               
# locationparam = f'{datapath}'+'/' + sys.argv[2] +'/' + sys.argv[4]
# # print( f'{datapath}'+'/' + sys.argv[2] +'/' + sys.argv[4])

# params_file = open(locationparam)

# varDict['LABEL'	]			= sys.argv[1]				
# varDict['DATAPATH']			= datapath							
# varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[2] +'/' 				
# varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[3] +'/'			

# varDict['PARCELS'] 		= varDict['INPUTFOLDER'] + sys.argv[5] #'skimTijd_new_REF.mtx' 		

# varDict['SKIMTIME'] 		= varDict['INPUTFOLDER'] + sys.argv[6] #'skimTijd_new_REF.mtx' 		
# varDict['SKIMDISTANCE']		= varDict['INPUTFOLDER'] + sys.argv[7] #'skimAfstand_new_REF.mtx'	
# varDict['ZONES']			= varDict['INPUTFOLDER'] + sys.argv[8] #'Zones_v4.shp'				
# varDict['SEGS']				= varDict['INPUTFOLDER'] + sys.argv[9] #'SEGS2020.csv'				
# varDict['PARCELNODES']		= varDict['INPUTFOLDER'] + sys.argv[10] #'parcelNodes_v2.shp'			

# # params_file = open(f'{datapath}/Input/Params_ParcelGen.txt')





# for line in params_file:
#     if len(line.split('=')) > 1:
#         key, value = line.split('=')
#         if len(value.split(':')) > 1:
#             value, dtype = value.split(':')
#             if len(dtype.split('#')) > 1: dtype, comment = dtype.split('#')
#             # Allow for spacebars around keys, values and dtypes
#             while key[0] == ' ' or key[0] == '\t': key = key[1:]
#             while key[-1] == ' ' or key[-1] == '\t': key = key[0:-1]
#             while value[0] == ' ' or value[0] == '\t': value = value[1:]
#             while value[-1] == ' ' or value[-1] == '\t': value = value[0:-1]
#             while dtype[0] == ' ' or dtype[0] == '\t': dtype = dtype[1:]
#             while dtype[-1] == ' ' or dtype[-1] == '\t': dtype = dtype[0:-1]
#             dtype = dtype.replace('\n',"")
#             # print(key, value, dtype)
#             if dtype == 'string': varDict[key] = str(value)
#             elif dtype == 'list': varDict[key] = ast.literal_eval(value)
#             elif dtype == 'int': varDict[key] = int(value)               
#             elif dtype == 'float': varDict[key] = float(value)               
#             elif dtype == 'bool': varDict[key] = eval(value)               
#             elif dtype == 'variable': varDict[key] = globals()[value]
#             elif dtype == 'eval': varDict[key] = eval(value)


# f"{varDict['OUTPUTFOLDER']}ParcelDemand_{varDict['LABEL']}.csv"


if sys.argv[0] == '':
    params_file = open(f'{datapath}/Input/Params_ParcelGen.txt')
    
    # This are the defaults, might need to change for console run!!!
    varDict['LABEL'	]			= 'ParcelLockers'				
    varDict['DATAPATH']			= datapath							
    varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ 'Input' +'/' 				
    varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ 'Output' +'/'			
    
    varDict['PARCELS']              = varDict['INPUTFOLDER'] + 'ParcelDemand_ParcelLockers.csv'     

    varDict['SKIMTIME'] 		= varDict['INPUTFOLDER'] + 'skimTijd_new_REF.mtx' #'skimTijd_new_REF.mtx' 		
    varDict['SKIMDISTANCE']		= varDict['INPUTFOLDER'] + 'skimAfstand_new_REF.mtx' #'skimAfstand_new_REF.mtx'	
    varDict['ZONES']			= varDict['INPUTFOLDER'] + 'Zones_v4.shp' #'Zones_v4.shp'				
    varDict['SEGS']				= varDict['INPUTFOLDER'] + 'SEGS2020.csv' #'SEGS2020.csv'				
    varDict['PARCELNODES']		= varDict['INPUTFOLDER'] + 'parcelNodes_v2.shp'				
    
    
    
    
else:  # This is the part for line cod execution
    locationparam = f'{datapath}'+'/' + sys.argv[2] +'/' + sys.argv[4]
    params_file = open(locationparam)

    varDict['LABEL'	]			= sys.argv[1]				
    varDict['DATAPATH']			= datapath							
    varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[2] +'/' 				
    varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[3] +'/'			
    
    varDict['PARCELS'] 		= varDict['INPUTFOLDER'] + sys.argv[5] #'skimTijd_new_REF.mtx' 		
    
    varDict['SKIMTIME'] 		= varDict['INPUTFOLDER'] + sys.argv[6] #'skimTijd_new_REF.mtx' 		
    varDict['SKIMDISTANCE']		= varDict['INPUTFOLDER'] + sys.argv[7] #'skimAfstand_new_REF.mtx'	
    varDict['ZONES']			= varDict['INPUTFOLDER'] + sys.argv[8] #'Zones_v4.shp'				
    varDict['SEGS']				= varDict['INPUTFOLDER'] + sys.argv[9] #'SEGS2020.csv'				
    varDict['PARCELNODES']		= varDict['INPUTFOLDER'] + sys.argv[10] #'parcelNodes_v2.shp'		
    # So it shuts up the warnings (remove when running in spyder)
    pd.options.mode.chained_assignment = None
   





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













#%% Start 
np.random.seed(int(varDict['Seed']))
zones = read_shape(varDict['ZONES'])
zones.index = zones['AREANR']
nZones = len(zones)

skims = {'time': {}, 'dist': {}, }
skims['time']['path'] = varDict['SKIMTIME']
skims['dist']['path'] = varDict['SKIMDISTANCE']
for skim in skims:
    skims[skim] = read_mtx(skims[skim]['path'])
    nSkimZones = int(len(skims[skim])**0.5)
    skims[skim] = skims[skim].reshape((nSkimZones, nSkimZones))
    if skim == 'time': skims[skim][6483] = skims[skim][:,6483] = 5000 # data deficiency
    for i in range(nSkimZones): #add traveltimes to internal zonal trips
        skims[skim][i,i] = 0.7 * np.min(skims[skim][i,skims[skim][i,:]>0])
skimTravTime = skims['time']; skimDist = skims['dist']
skimDist_flat = skimDist.flatten()
del skims, skim, i
    
zoneDict  = dict(np.transpose(np.vstack( (np.arange(1,nZones+1), zones['AREANR']) )))
zoneDict  = {int(a):int(b) for a,b in zoneDict.items()}
invZoneDict = dict((v, k) for k, v in zoneDict.items()) 

segs   = pd.read_csv(varDict['SEGS'])
segs.index = segs['zone']
segs = segs[segs['zone'].isin(zones['AREANR'])] #Take only segs into account for which zonal data is known as well

parcelNodesPath = varDict['PARCELNODES']
parcelNodes = read_shape(parcelNodesPath, returnGeometry=False)
parcelNodes.index   = parcelNodes['id'].astype(int)
parcelNodes         = parcelNodes.sort_index()    

for node in parcelNodes['id']:
    parcelNodes.loc[node,'SKIMNR'] = int(invZoneDict[parcelNodes.at[int(node),'AREANR']])
parcelNodes['SKIMNR'] = parcelNodes['SKIMNR'].astype(int)

cepList   = np.unique(parcelNodes['CEP'])
cepNodes = [np.where(parcelNodes['CEP']==str(cep))[0] for cep in cepList]

cepNodeDict = {}; cepZoneDict = {}; cepSkimDict = {}
for cep in cepList: 
    cepZoneDict[cep] = parcelNodes[parcelNodes['CEP'] == cep]['AREANR'].astype(int).tolist()
    cepSkimDict[cep] = parcelNodes[parcelNodes['CEP'] == cep]['SKIMNR'].astype(int).tolist()
for cepNo in range(len(cepList)):
    cepNodeDict[cepList[cepNo]] = cepNodes[cepNo]

    
    
    




parcels = pd.read_csv(varDict['PARCELS'])  # Load the output of PARCEL_DMND


parcels.index = parcels['Parcel_ID']
parcels['Segment'] = np.where(np.random.uniform(0,1,len(parcels)) < (varDict['PARCELS_PER_HH_C2C']/varDict['PARCELS_PER_HH']), 'C2C', 'B2C')

parcels['L2L'] = np.random.uniform(0,1,len(parcels)) < varDict['Local2Local'] #make certain percentage L2L
parcels['CS_eligible'] = np.random.uniform(0,1,len(parcels)) < varDict['CS_cust_willingness'] # make certain percentage CS eligible. This eligibility is A PRIORI, depending on parcel/sender/receiver characteristics. Inside the crowdshipping part we might want to adjust that according to a choice model
parcels['CS_eligible'] = (parcels['CS_eligible'] & parcels['L2L'] ) # This means that the CS parcels are only L2L and the percentage above is the % of the L2L parcels that can be crowdshipped



parcels_hyperconnected = parcels[parcels['L2L'] | parcels['CS_eligible']   ]




Gemeenten = varDict['Gemeenten_studyarea']

if len(Gemeenten) > 1:  # If there are more than 1 gemente in the list
    ParceltobeL2L = pd.DataFrame(columns = parcels_hyperconnected.columns)

    for Geemente in Gemeenten:
        if type (Geemente) != list: # If there the cities are NOT connected (that is every geemente is separated from the next)
            ParcelTemp = parcels_hyperconnected[parcels_hyperconnected['D_zone'].isin(zones['AREANR'][zones['GEMEENTEN']==Geemente])]
            origin_distribution = segs['1: woningen']/segs['1: woningen'].sum() #create distribution to number of houses
            ParcelTemp['O_zone'] = np.random.choice(segs['zone'], p=(origin_distribution), size=(len(ParcelTemp))) 
            segs_local = segs[segs['zone'].isin(zones['AREANR'][zones['GEMEENTEN']==Geemente])] #only consider SEGS in study area for L2L
            L2L_distribution = segs_local['1: woningen']/segs_local['1: woningen'].sum() #create distribution to number of houses in area
            ParcelTemp.loc[ParcelTemp['L2L'] == True,'O_zone'] = np.random.choice(segs_local['zone'], p=(L2L_distribution), size=( ParcelTemp['L2L'].value_counts()[True] )) #assign origin based on local distribution if parcel is L2L
            ParceltobeL2L = ParceltobeL2L.append(ParcelTemp)
        else:
            ParcelTemp = parcels_hyperconnected[parcels_hyperconnected['D_zone'].isin(zones['AREANR'][zones['GEMEENTEN'].isin(Geemente)])]
            origin_distribution = segs['1: woningen']/segs['1: woningen'].sum() #create distribution to number of houses
            ParcelTemp['O_zone'] = np.random.choice(segs['zone'], p=(origin_distribution), size=(len(ParcelTemp))) 
            segs_local = segs[segs['zone'].isin(zones['AREANR'][zones['GEMEENTEN'].isin(Geemente)])]#only consider SEGS in study area for L2L
            L2L_distribution = segs_local['1: woningen']/segs_local['1: woningen'].sum() #create distribution to number of houses in area
            ParcelTemp.loc[ParcelTemp['L2L'] == True,'O_zone'] = np.random.choice(segs_local['zone'], p=(L2L_distribution), size=( ParcelTemp['L2L'].value_counts()[True] )) #assign origin based on local distribution if parcel is L2L
            ParceltobeL2L = ParceltobeL2L.append(ParcelTemp)
        
    parcels_hyperconnected = ParceltobeL2L
  
else:    # print(len(ParceltobeL2L))
    if type (Gemeenten[0]) == list:
        Geemente = Gemeenten [0]
    else:
        Geemente = Gemeenten
    parcels_hyperconnected = parcels_hyperconnected[parcels_hyperconnected['D_zone'].isin(zones['AREANR'][zones['GEMEENTEN'].isin(Geemente)])] #filter the parcels to study area
    origin_distribution = segs['1: woningen']/segs['1: woningen'].sum() #create distribution to number of houses
    parcels_hyperconnected['O_zone'] = np.random.choice(segs['zone'], p=(origin_distribution), size=(len(parcels_hyperconnected))) #assign origin based on distribution  
    segs_local = segs[segs['zone'].isin(zones['AREANR'][zones['GEMEENTEN'].isin(Geemente)])] #only consider SEGS in study area for L2L
    L2L_distribution = segs_local['1: woningen']/segs_local['1: woningen'].sum() #create distribution to number of houses in area
    parcels_hyperconnected.loc[parcels_hyperconnected['L2L'] == True,'O_zone'] = np.random.choice(segs_local['zone'], p=(L2L_distribution), size=( parcels_hyperconnected['L2L'].value_counts()[True] )) #assign origin based on local distribution if parcel is L2L
    



ParcelLockers =  parcels[ (parcels['PL']!=0 ) ]  # and  ! parcels['L2L'] and ! parcels['CS_eligible']  ) ]  
ParcelLockers = ParcelLockers [  (parcels['L2L']== False )           ]
ParcelLockers = ParcelLockers [  (parcels['CS_eligible']== False )           ]


parcels_hyperconnected = parcels_hyperconnected.append(ParcelLockers)                                               

'''B2C parcels generation''' # This is actually hub and     # Why are we doing this? This would be L2L and B2B only, not the hubspoke parcels from outside
parcels_hubspoke = parcels[~(parcels['L2L'] | parcels['CS_eligible']| parcels['PL'])]

'''

I took this part away because it was generating some origins that we don't need now. This can be reviewed once we separate segments B2B from C2C and B2C
The problem here is that the origin is in the study area, which is not real

parcels_hubspoke = parcels_hubspoke.rename(columns={'O_zone': 'D_DepotZone', 'DepotNumber': 'D_DepotNumber'}) #change the column names
origin_distribution = segs['6: detail']/segs['6: detail'].sum() #initiate origin distribution based on retail jobs in the zones
parcels_hubspoke['O_zone'] = np.random.choice(segs['zone'], p=(origin_distribution), size=(len(parcels_hubspoke))) #apply distribution
parcels_hubspoke['Segment'] = 'B2C'
parcels_hubspoke['O_DepotZone'], parcels_hubspoke['O_DepotNumber'] = '', ''
for index in parcels_hubspoke.index:
    cep = parcels_hubspoke.at[index, 'CEP']
    o_zone = parcels_hubspoke.at[index, 'O_zone']
    parcels_hubspoke.at[index, 'O_DepotZone'] = cepZoneDict[cep][skimTravTime[invZoneDict[o_zone]-1,[x-1 for x in cepSkimDict[cep]]].argmin()] #search closest depot zone to origin
    parcels_hubspoke.at[index, 'O_DepotNumber'] = cepNodeDict[cep][skimTravTime[invZoneDict[o_zone]-1,[x-1 for x in cepSkimDict[cep]]].argmin()]+1 #get depot number of this depot
parcels_hubspoke = parcels_hubspoke[['Parcel_ID', 'O_zone', 'O_DepotZone', 'D_DepotZone', 'D_zone', 'O_DepotNumber', 'D_DepotNumber', 'CEP', 'VEHTYPE', 'Segment']] #rearrange columns
'''

parcels_hyperconnected ['Fulfilment'] = 'Hyperconnected'
parcels_hubspoke['Fulfilment'] = 'Hubspoke'
0

Parcels = parcels_hubspoke.append(parcels_hyperconnected)


# parcels.to_csv(f"{varDict['OUTPUTFOLDER']}Demand_parcels_{varDict['LABEL']}.csv", index=False)
# parcels_hyperconnected.to_csv(f"{varDict['OUTPUTFOLDER']}Demand_parcels_hyperconnected{varDict['LABEL']}.csv", index=False)
# parcels_hubspoke.to_csv(f"{varDict['OUTPUTFOLDER']}Demand_parcels_hubspoke{varDict['LABEL']}.csv", index=False)

Parcels.to_csv(f"{varDict['OUTPUTFOLDER']}Demand_parcels_fulfilment_{varDict['LABEL']}.csv", index=False)
    
print("Connection Generation 2 Market done")

    #%% Recover previous run if not generated the demand
    
    # if ~ varDict['RUN_DEMAND_MODULE']:
    #     print("Get parcel demand from previous run...")
    #     parcels = pd.read_csv(f"{varDict['DATAPATH']}Output/Demand_parcels.csv");    parcels.index = parcels['Parcel_ID']
    #     parcels_hyperconnected = pd.read_csv(f"{varDict['DATAPATH']}Output/Demand_parcels_hyperconnected.csv"); parcels_hyperconnected.index = parcels_hyperconnected['Parcel_ID']
    #     parcels_hubspoke = pd.read_csv(f"{varDict['DATAPATH']}Output/Demand_parcels_hubspoke.csv"); parcels_hubspoke.index = parcels_hubspoke['Parcel_ID']
    
    
    #%%
"""

The outputs are 2 datasets:
    1 The parels that go to the ParcelMarket
    2 For the parcels that go through hub_spoke. The delivery and pick up will be changed outside
    

"""

























