# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 14:08:07 2022

@author: rtapia
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:47:07 2021

@author: rtapia
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 08:56:07 2021

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
datapath = cwd.replace('Code', '')

#%% Define all variables
def generate_args(method):
    varDict = {}
   
    '''FOR ALL MODULES'''
    cwd = os.getcwd().replace(os.sep, '/')
    datapath = cwd.replace('Code', '')
    
    if method == 'from_file':   
            
        if sys.argv[0] == '':
            params_file = open(f'{datapath}/Input/Params_ParcelSched.txt')
            
            varDict['LABEL'	]			= 'MVP1c'			
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ 'Input'+'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ 'Output' +'/'		
            
            varDict['Parcels'] 		    = varDict['INPUTFOLDER']   + 'ParcelDemand_MVP1c.csv' 	
            varDict['Parcels_Hub2Hub']  = varDict['INPUTFOLDER'] + 'ParcelDemand_Hub2Hub_MVP1c.csv' 	
            varDict['SKIMTIME'] 		= varDict['INPUTFOLDER']     + 'skimTijd_new_REF.mtx' 		
            varDict['SKIMDISTANCE']	    = varDict['INPUTFOLDER'] + 'skimAfstand_new_REF.mtx'	
            varDict['ZONES']			= varDict['INPUTFOLDER']       + 'Zones_v4.shp'				
            varDict['SEGS']				= varDict['INPUTFOLDER']       + 'SEGS2020.csv'				
            varDict['PARCELNODES']		= varDict['INPUTFOLDER']   + 'parcelNodesv2CycloonMyPup.shp'				
            varDict['CROWDSHIPPING']		= False # This is the false for the crowdshipping module within the scheduling. It is NOT the LEAD CS module.			
            varDict["DepartureCDF"]    =   varDict['INPUTFOLDER']   +  "departureTimeParcelsCDF.csv"
            
            
            
        else:  # This is the part for line cod execution
            locationparam = f'{datapath}' + '/' + sys.argv[2] +'/' + sys.argv[4]
            params_file = open(locationparam)
            varDict['LABEL'	]			= sys.argv[1]				
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[2] +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[3] +'/'		
            
            varDict['Parcels'] 		    = varDict['INPUTFOLDER'] + sys.argv[5] #'parcel demand csv' 	
            varDict['Parcels_Hub2Hub']  = varDict['INPUTFOLDER'] + sys.argv[6] #'parcel demand csv' 	
            varDict['SKIMTIME'] 		= varDict['INPUTFOLDER'] + sys.argv[7] #'skimTijd_new_REF.mtx' 		
            varDict['SKIMDISTANCE']	    = varDict['INPUTFOLDER'] + sys.argv[8] #'skimAfstand_new_REF.mtx'	
            varDict['ZONES']			= varDict['INPUTFOLDER'] + sys.argv[9] #'Zones_v4.shp'				
            varDict['SEGS']				= varDict['INPUTFOLDER'] + sys.argv[10] #'SEGS2020.csv'				
            varDict['PARCELNODES']		= varDict['INPUTFOLDER'] + sys.argv[11] #'parcelNodes_v2.shp'				
            varDict['CROWDSHIPPING']		= False # This is the false for the crowdshipping module within the scheduling. It is NOT the LEAD CS module.			
            pd.options.mode.chained_assignment = None
            varDict["DepartureCDF"]    =   varDict['INPUTFOLDER']   + sys.argv[12]   # ADD TO JSON INPUT AND COMAND LINE

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
            
            
            
            
            
            
    elif method == 'from_code':
        print('Generating args from code')
        varDict['RUN_DEMAND_MODULE']            = False
        varDict['CROWDSHIPPING_NETWORK']        = True
        varDict['COMBINE_DELIVERY_PICKUP_TOUR'] = True
        varDict['HYPERCONNECTED_NETWORK']       = True
        
        varDict['LABEL']                = 'C2C'
        varDict['DATAPATH']             = datapath + '/'
        varDict['INPUTFOLDER']          = varDict['DATAPATH']+'Input/'
        varDict['OUTPUTFOLDER']         = varDict['DATAPATH']+'Output/'
        # varDict['PARAMFOLDER']	        = f'{datapath}Parameters/Mass-GT/'
        
        varDict['SKIMTIME']             = varDict['INPUTFOLDER'] + 'skimTijd_new_REF.mtx'
        varDict['SKIMDISTANCE']         = varDict['INPUTFOLDER'] + 'skimAfstand_new_REF.mtx'
        varDict['ZONES']                = varDict['INPUTFOLDER'] + 'Zones_v4.shp'
        varDict['SEGS']                 = varDict['INPUTFOLDER'] + 'SEGS2020.csv'
        varDict['PARCELNODES']          = varDict['INPUTFOLDER'] + 'parcelNodes_v2.shp'
        varDict['CEP_SHARES']           = varDict['INPUTFOLDER'] + 'CEPshares.csv'
        varDict['Pax_Trips']            = varDict['INPUTFOLDER'] + 'trips.csv'
        
        
        '''FOR PARCEL MARKET MODULE'''
        varDict['hub_zones']                  = [585]
        varDict['parcelLockers_zones']        = [585]
        varDict['Gemeenten_studyarea']  = [ 
                                            [ 'Delft', 'Midden_Delfland', 'Rijswijk','sGravenhage','Leidschendam_Voorburg',],
                                            # [ 'Rotterdam','Schiedam','Vlaardingen','Ridderkerk', 'Barendrecht',],
                                            
            
                                            # 'Albrandswaard',
                                            # #     'Barendrecht',#
                                            #     'Brielle',
                                            #     'Capelle aan den IJssel',
                                            # #     'Delft', #
                                            #     'Hellevoetsluis',
                                            #     'Krimpen aan den IJssel',
                                            #     'Lansingerland',
                                            # #     'Leidschendam_Voorburg',#
                                            #     'Maassluis',
                                            # #     'Midden_Delfland',#
                                            #     'Nissewaard',
                                            #     'Pijnacker_Nootdorp',
                                            # #     'Ridderkerk',#
                                            # #     'Rijswijk',#
                                            # #     'Rotterdam',#
                                            # #     'Schiedam',#
                                            # #     'Vlaardingen',#
                                            #     'Wassenaar',
                                            #     'Westland',
                                            #     'Westvoorne',
                                            #     'Zoetermeer',
                                            #     # 'sGravenhage'#
                                              ]
        # Hague
        varDict['Gemeenten_CS']         = ["sGravenhage", "Zoetermeer", "Midden_Delfland"]

          
        '''FOR PARCEL SCHEDULING MODULE'''
        varDict['PARCELS_MAXLOAD']      = 180
        varDict['PARCELS_DROPTIME']     = 120
        varDict['PARCELS_SUCCESS_B2C']  = 0.75
        varDict['PARCELS_SUCCESS_B2B']  = 0.95
        varDict['PARCELS_GROWTHFREIGHT']= 1.0
        varDict['CROWDSHIPPING']        = False #SCHED module has own CS integrated, this is not used here
        varDict['CRW_PARCELSHARE']      = 0.1
        varDict['CRW_MODEPARAMS']       = varDict['INPUTFOLDER'] + 'Params_UseCase_CrowdShipping.csv'
        

    
    
    args = ['', varDict]
    return args, varDict

method = 'from_file' #either from_file or from_code
args, varDict = generate_args(method)

# TESTRUN = False # True to fasten further code TEST (runs with less parcels)
# TestRunLen = 100










#%%


#%% Module 0: Load input data
'''
These variables will be used throughout the whole model
'''

Comienzo = dt.datetime.now()
print ("Comienzo: ",Comienzo)



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

KPIs = {}

#%%










#%%







# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:21:12 2020
@author: modelpc
"""
import numpy as np
import pandas as pd
import time
import datetime
from __functions__ import read_mtx, read_shape

# Modules nodig voor de user interface
import tkinter as tk
from tkinter.ttk import Progressbar
import zlib
import base64
import tempfile
from threading import Thread



def main(varDict):
    '''
    Start the GUI object which runs the module
    '''
    root = Root(varDict)
    
    return root.returnInfo
    


#%% Class: Root

class Root:
    
    def __init__(self, args):       
        '''
        Initialize a GUI object
        '''        
        # Set graphics parameters
        self.width  = 500
        self.height = 60
        self.bg     = 'black'
        self.fg     = 'white'
        self.font   = 'Verdana'
        
        # Create a GUI window
        self.root = tk.Tk()
        self.root.title("Progress Parcel Scheduling")
        self.root.geometry(f'{self.width}x{self.height}+0+200')
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg=self.bg)
        self.canvas.place(x=0, y=0)
        self.statusBar = tk.Label(self.root, text="", anchor='w', borderwidth=0, fg='black')
        self.statusBar.place(x=2, y=self.height-22, width=self.width, height=22)
        
        # Remove the default tkinter icon from the window
        icon = zlib.decompress(base64.b64decode('eJxjYGAEQgEBBiDJwZDBy''sAgxsDAoAHEQCEGBQaIOAg4sDIgACMUj4JRMApGwQgF/ykEAFXxQRc='))
        _, self.iconPath = tempfile.mkstemp()
        with open(self.iconPath, 'wb') as iconFile:
            iconFile.write(icon)
        self.root.iconbitmap(bitmap=self.iconPath)
        
        # Create a progress bar
        self.progressBar = Progressbar(self.root, length=self.width-20)
        self.progressBar.place(x=10, y=10)
        
        self.returnInfo = ""
        
        if __name__ == '__main__':
            self.args = [[self, args]]
        else:
            self.args = [args]
        
        self.run_module()       
        
        # Keep GUI active until closed    
        self.root.mainloop()
        
        
        
    def update_statusbar(self, text):
        self.statusBar.configure(text=text)



    def error_screen(self, text='', event=None, size=[800,50], title='Error message'):
        '''
        Pop up a window with an error message
        '''
        windowError = tk.Toplevel(self.root)
        windowError.title(title)
        windowError.geometry(f'{size[0]}x{size[1]}+0+{200+50+self.height}')
        windowError.minsize(width=size[0], height=size[1])
        windowError.iconbitmap(default=self.iconPath)
        labelError = tk.Label(windowError, text=text, anchor='w', justify='left')
        labelError.place(x=10, y=10)  
        
        

    def run_module(self, event=None):
        Thread(target=actually_run_module, args=self.args, daemon=True).start()
        


#%% Function: actually_run_module
        
def actually_run_module(args):


       
    start_time = time.time()
    
   
    root    = args[0]
    varDict = args[1]
    

    if root != '':
        root.progressBar['value'] = 0
            
    # Define folders relative to current datapath
    datapathI = varDict['INPUTFOLDER']
    datapathO = varDict['OUTPUTFOLDER']
    # datapathP = varDict['PARAMFOLDER']
    zonesPath        = varDict['ZONES']
    skimTravTimePath = varDict['SKIMTIME']
    skimDistancePath = varDict['SKIMDISTANCE'] 
    parcelNodesPath  = varDict['PARCELNODES']
    segsPath         = varDict['SEGS']
    label            = varDict['LABEL']
    
    
    
    if varDict['Type'] == 'Hub2Hub':
    
        # dropOffTimeSec = varDict['PARCELS_DROPTIME_Hub2Hub']
        # maxVehicleLoad = varDict['PARCELS_MAXLOAD_Hub2Hub']
        # maxVehicleLoad = int(maxVehicleLoad)
        ParcelFile     =  varDict['Parcels_Hub2Hub']
        label          = label + '_Hub2Hub'
    
    elif varDict['Type'] == 'LastMile':
        # dropOffTimeSec = varDict['PARCELS_DROPTIME']
        # maxVehicleLoad = varDict['PARCELS_MAXLOAD']
        # maxVehicleLoad = int(maxVehicleLoad)
        ParcelFile     =  varDict['Parcels']
        label          =  label + '_LastMile'        
        
    elif varDict['Type'] == 'CombinedConsol_LastMile':
        ParcelFileL2L     =  varDict['Parcels']
        ParcelFileH2H     = varDict['Parcels_Hub2Hub']
        label          =  label + '_TotalUrbanDelivery'    
        for i,v in enumerate(varDict["VEHICLES"]):
            if varDict["VEHICLES"][v][0] !=  varDict["VEHICLES"][v][1]:
                print("Warning: Consolidated parcel trips between hubs will not be combined with last mile delivery for ",v," because the vehicles indicated are different!")
        
    
    
    print('Type ', varDict['Type'])
    # print('DropTime ',dropOffTimeSec) 
    # print('maxVehicleLoad, ', maxVehicleLoad)  
    # print('Parcel file: ',ParcelFile)
    
    
    
    
    doCrowdShipping = (str(varDict['CROWDSHIPPING']).upper() == 'TRUE')
    
    exportTripMatrix = True
    
    log_file = open(datapathO + "Logfile_ParcelScheduling.log", "w")
    log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
    
    if root != '':
        root.progressBar['value'] = 0.1
    
    
    # --------------------------- Import data----------------------------------
    print('Importing data...'), log_file.write('Importing data...\n')

    if varDict['Type'] == 'CombinedConsol_LastMile':
        parcelL2L = pd.read_csv(ParcelFileL2L)
        parcelL2L ["Type"] = "tour-based"
        parcelL2L ["Network"] = "conventional"
        parcelH2H = pd.read_csv(ParcelFileH2H)
        parcelH2H  ["Task"] = "Delivery" # Or should I put consolidation?
        parcelH2H  ["VEHTYPE"] = 7
        
        
        parcels = parcelL2L.append(parcelH2H)
        parcels= parcels.reset_index(drop=True)
        
    else:
        parcels = pd.read_csv(ParcelFile)
    
    
    
    
    
    
    
    parcelNodes, coords = read_shape(parcelNodesPath, returnGeometry=True)
    parcelNodes['X']    = [coords[i]['coordinates'][0] for i in range(len(coords))]
    parcelNodes['Y']    = [coords[i]['coordinates'][1] for i in range(len(coords))]
    '''EDIT 6-10: SHADOW PARCEL NODES - start'''
    parcelNodes_shadow = parcelNodes.copy()
    parcelNodes_shadow['id'] = parcelNodes_shadow['id']+1000
    parcelNodes = parcelNodes.append(parcelNodes_shadow, ignore_index=True,sort=False)
    '''EDIT 6-10: SHADOW PARCEL NODES - end'''
    parcelNodes['id'] = parcelNodes['id'].astype(int)
    parcelNodes.index = parcelNodes['id']
    parcelNodes = parcelNodes.sort_index()    
    parcelNodesCEP = {}
    for i in parcelNodes.index:
        parcelNodesCEP[parcelNodes.at[i,'id']] = parcelNodes.at[i,'CEP']
    
    zones = read_shape(zonesPath)
    zones = zones.sort_values('AREANR')
    zones.index = zones['AREANR']
    supCoordinates = pd.read_csv(datapathI + 'SupCoordinatesID.csv', sep=',')
    supCoordinates.index = supCoordinates['AREANR']
    
    zonesX = {}
    zonesY = {}
    for areanr in zones.index:
        zonesX[areanr] = zones.at[areanr, 'X']
        zonesY[areanr] = zones.at[areanr, 'Y']
    for areanr in supCoordinates.index:
        zonesX[areanr] = supCoordinates.at[areanr, 'Xcoor']
        zonesY[areanr] = supCoordinates.at[areanr, 'Ycoor']                    
    
    nIntZones = len(zones)
    nSupZones = 43
    zoneDict  = dict(np.transpose(np.vstack( (np.arange(1,nIntZones+1), zones['AREANR']) )))
    zoneDict  = {int(a):int(b) for a,b in zoneDict.items()}
    for i in range(nSupZones):
        zoneDict[nIntZones+i+1] = 99999900 + i + 1
    invZoneDict = dict((v, k) for k, v in zoneDict.items())
    
    # Change zoning to skim zones which run continuously from 0
    # parcels = parcels.dropna(subset=['D_zone'])
    parcels['X'] = [zonesX[x] for x in parcels['D_zone'].values]
    parcels['Y'] = [zonesY[x] for x in parcels['D_zone'].values]
    parcels['D_zone']        = [invZoneDict[x] for x in parcels['D_zone']]
    parcels['O_zone']        = [invZoneDict[x] for x in parcels['O_zone']]
    parcelNodes['skim_zone'] = [invZoneDict[x] for x in parcelNodes['AREANR']]
    
    if root != '':
        root.progressBar['value'] = 0.3
        
    # System input for scheduling
    parcelDepTime = np.array(pd.read_csv(varDict["DepartureCDF"]).iloc[:,1])        #f"{datapathI}departureTimeParcelsCDF.csv"
    # dropOffTime   = dropOffTimeSec/3600    
    skimTravTime  = read_mtx(skimTravTimePath)
    skimDistance  = read_mtx(skimDistancePath)
    nZones        = int(len(skimTravTime)**0.5)
    
    if root != '':
        root.progressBar['value'] = 0.9
        
    # Intrazonal impedances
    skimTravTime = skimTravTime.reshape(nZones,nZones)
    skimTravTime[:,6483] = 2000.0
    for i in range(nZones):
        skimTravTime[i,i] = 0.7 * np.min(skimTravTime[i,skimTravTime[i,:]>0])
    skimTravTime = skimTravTime.flatten()
    skimDistance = skimDistance.reshape(nZones,nZones)
    for i in range(nZones):
        skimDistance[i,i] = 0.7 * np.min(skimDistance[i,skimDistance[i,:]>0])
    skimDistance = skimDistance.flatten()
    
    depotIDs   = list(parcelNodes['id'])
    
    if root != '':
        root.progressBar['value'] = 1.0
        
    
    # ------------------- Crowdshipping use case -----------------------------------
    # if doCrowdShipping:
        
    #     # The first N zones for which to consider crowdshipping for parcel deliveries
    #     nFirstZonesCS  = 5925
        
    #     # Percentage of parcels eligible for crowdshipping
    #     parcelShareCRW = varDict['CRW_PARCELSHARE']
        
    #     # Input data and parameters for the crowdshipping use case
    #     modeParamsCRW = pd.read_csv(varDict['CRW_MODEPARAMS'], index_col=0, sep=',')
        
    #     modes = {'fiets': {}, 'auto': {}, }
    #     modes['fiets']['willingness' ] = modeParamsCRW.at['BIKE','WILLINGNESS']
    #     modes['auto' ]['willingness' ] = modeParamsCRW.at['CAR', 'WILLINGNESS']
    #     modes['fiets']['dropoff_time'] = modeParamsCRW.at['BIKE','DROPOFFTIME']
    #     modes['auto' ]['dropoff_time'] = modeParamsCRW.at['CAR', 'DROPOFFTIME']
    #     modes['fiets']['VoT'         ] = modeParamsCRW.at['BIKE','VOT']            
    #     modes['auto' ]['VoT'         ] = modeParamsCRW.at['CAR', 'VOT']
        
    #     modes['fiets']['relative_extra_parcel_dist_threshold'] = modeParamsCRW.at['BIKE','RELATIVE_EXTRA_PARCEL_DIST_THRESHOLD']
    #     modes['auto' ]['relative_extra_parcel_dist_threshold'] = modeParamsCRW.at['CAR', 'RELATIVE_EXTRA_PARCEL_DIST_THRESHOLD']
        
    #     modes['fiets']['OD_path'  ] = varDict['CRW_PDEMAND_BIKE']
    #     modes['fiets']['skim_time'] = skimDistance / 1000 / 12 * 3600
    #     modes['fiets']['n_trav'   ] = 0
    #     modes['auto']['OD_path'  ] = varDict['CRW_PDEMAND_CAR']
    #     modes['auto']['skim_time'] = skimTravTime
    #     modes['auto']['n_trav'   ] = 0            
        
    #     for mode in modes:
    #         modes[mode]['OD_array']  = read_mtx(modes[mode]['OD_path']).reshape(nIntZones, nIntZones)
    #         modes[mode]['OD_array'] *= modes[mode]['willingness']
    #         modes[mode]['OD_array']  = np.round(modes[mode]['OD_array'], 0)
    #         modes[mode]['OD_array']  = np.array(modes[mode]['OD_array'], dtype=int)
    #         modes[mode]['OD_array']  = modes[mode]['OD_array'][:nFirstZonesCS,:]
    #         modes[mode]['OD_array']  = modes[mode]['OD_array'][:,:nFirstZonesCS]
            
    #     # Which zones are located in which municipality
    #     zone_gemeente_dict = dict(np.transpose(np.vstack( (np.arange(1,nIntZones+1), zones['Gemeentena']) )))
        
    #     # Get retail jobs per zone from socio-economic data
    #     segs       = pd.read_csv(segsPath)
    #     segs.index = segs['zone']
    #     segsDetail = np.array(segs['6: detail'])
        
    #     # Perform the crowdshipping calculations
    #     do_crowdshipping(parcels, zones, nIntZones, nZones, zoneDict, zonesX, zonesY, 
    #                      skimDistance, skimTravTime, 
    #                      nFirstZonesCS, parcelShareCRW, modes, zone_gemeente_dict, segsDetail,
    #                      datapathO, label, log_file,
    #                      root)
    
    
    # ----------------------- Forming spatial clusters of parcels -----------------        
    print('Forming spatial clusters of parcels...')
    log_file.write('Forming spatial clusters of parcels...\n')
    
    # A measure of euclidean distance based on the coordinates
    skimEuclidean  = (np.array(list(zonesX.values())).repeat(nZones).reshape(nZones,nZones) -
                      np.array(list(zonesX.values())).repeat(nZones).reshape(nZones,nZones).transpose())**2
    skimEuclidean += (np.array(list(zonesY.values())).repeat(nZones).reshape(nZones,nZones) -
                      np.array(list(zonesY.values())).repeat(nZones).reshape(nZones,nZones).transpose())**2
    skimEuclidean = skimEuclidean**0.5
    skimEuclidean = skimEuclidean.flatten()
    skimEuclidean /= np.sum(skimEuclidean)
    
    # To prevent instability related to possible mistakes in skim, 
    # use average of skim and euclidean distance (both normalized to a sum of 1)
    skimClustering  = skimDistance.copy()
    skimClustering = skimClustering / np.sum(skimClustering)
    skimClustering += skimEuclidean
    
    del skimEuclidean
    
    if label == 'UCC':
        
        # Divide parcels into the 4 tour types, namely:
        # 0: Depots to households
        # 1: Depots to UCCs
        # 2: From UCCs, by van
        # 3: From UCCs, by LEVV
        parcelsUCC = {}
        parcelsUCC[0] = pd.DataFrame(parcels[(parcels['FROM_UCC']==0) & (parcels['TO_UCC']==0)])
        parcelsUCC[1] = pd.DataFrame(parcels[(parcels['FROM_UCC']==0) & (parcels['TO_UCC']==1)])
        parcelsUCC[2] = pd.DataFrame(parcels[(parcels['FROM_UCC']==1) & (parcels['VEHTYPE']==7)])
        parcelsUCC[3] = pd.DataFrame(parcels[(parcels['FROM_UCC']==1) & (parcels['VEHTYPE']==8)])
    
        # Cluster parcels based on proximity and constrained by vehicle capacity
        for i in range(3):
            if doCrowdShipping:                    
                startValueProgress = 56.0 +     i/3 * (70.0 - 56.0)
                endValueProgress   = 56.0 + (i+1)/3 * (70.0 - 56.0)
            else:
                startValueProgress = 2.0 +     i/3 * (55.0 - 2.0)
                endValueProgress   = 2.0 + (i+1)/3 * (55.0 - 2.0)
            print('\tTour type ' + str(i+1) + '...'), log_file.write('\tTour type ' + str(i+1) + '...\n')
            parcelsUCC[i] = cluster_parcels(parcelsUCC[i], maxVehicleLoad, skimClustering,
                                            root, startValueProgress, endValueProgress)
    
        # LEVV have smaller capacity
        startValueProgress = 70.0 if doCrowdShipping else 55.0
        startValueProgress = 75.0 if doCrowdShipping else 60.0
        print('\tTour type 4...'), log_file.write('\tTour type 4...\n')
        parcelsUCC[3] = cluster_parcels(parcelsUCC[3], int(round(maxVehicleLoad/5)), skimClustering,
                                        root, startValueProgress, endValueProgress)
    
        # Aggregate parcels based on depot, cluster and destination
        for i in range(4):
            if i <= 1:
                parcelsUCC[i] = pd.pivot_table(parcelsUCC[i], 
                                               values=['Parcel_ID'], 
                                               index=['DepotNumber', 'Cluster', 'O_zone', 'D_zone'], 
                                               aggfunc = {'Parcel_ID': 'count'})           
                parcelsUCC[i] = parcelsUCC[i].rename(columns={'Parcel_ID':'Parcels'}) 
                parcelsUCC[i]['Depot'  ] = [x[0] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Cluster'] = [x[1] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Orig'   ] = [x[2] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Dest'   ] = [x[3] for x in parcelsUCC[i].index]
            else:
                parcelsUCC[i] = pd.pivot_table(parcelsUCC[i], 
                                               values=['Parcel_ID'], 
                                               index=['O_zone', 'Cluster', 'D_zone'], 
                                               aggfunc = {'Parcel_ID': 'count'})                      
                parcelsUCC[i] = parcelsUCC[i].rename(columns={'Parcel_ID':'Parcels'}) 
                parcelsUCC[i]['Depot'  ] = [x[0] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Cluster'] = [x[1] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Orig'   ] = [x[0] for x in parcelsUCC[i].index]
                parcelsUCC[i]['Dest'   ] = [x[2] for x in parcelsUCC[i].index]
            parcelsUCC[i].index = np.arange(len(parcelsUCC[i]))
               
    if label != 'UCC':
        # Cluster parcels based on proximity and constrained by vehicle capacity
        startValueProgress = 56.0 if doCrowdShipping else 2.0
        endValueProgress   = 75.0 if doCrowdShipping else 60.0
        
        
        parcels["VEHTYPE"] = parcels["CEP"].map(varDict["VEHICLES"])
        
        if varDict['Type'] == "LastMile":
            parcels["VEHTYPE"] = parcels["VEHTYPE"].str[0]  
        if varDict['Type'] == "Hub2Hub":
            parcels["VEHTYPE"] = parcels["VEHTYPE"].str[1]
        if varDict['Type'] == 'CombinedConsol_LastMile':
               parcels["VEHTYPE"] = np.where(parcels["Type"] == "tour-based", parcels["VEHTYPE"].str[0] , np.where ( parcels["Type"] == "consolidated",parcels["VEHTYPE"].str[1],"NA"))
        
        
        parcels = cluster_parcels(parcels, varDict["CAPACITY"], skimClustering,
                                  root, startValueProgress, endValueProgress)
        
        # Aggregate parcels based on depot, cluster and destination
        parcels = pd.pivot_table(parcels, 
                                 values=['Parcel_ID'], 
                                 index=['DepotNumber', 'Cluster', 'O_zone', 'D_zone',"VEHTYPE"], 
                                 aggfunc = {'Parcel_ID': 'count'})           
        parcels = parcels.rename(columns={'Parcel_ID':'Parcels'}) 
        parcels['Depot'  ] = [x[0] for x in parcels.index]
        parcels['Cluster'] = [x[1] for x in parcels.index]
        parcels['Orig'   ] = [x[2] for x in parcels.index]
        parcels['Dest'   ] = [x[3] for x in parcels.index]
        parcels['VEHTYPE'] = [x[4] for x in parcels.index]
        parcels.index = np.arange(len(parcels))
        
    
    del skimClustering
    
    
    # ----------- Scheduling of trips (UCC scenario) --------------------------
    
    if label == 'UCC':
            
        # Depots to households
        print('Starting scheduling procedure for parcels from depots to households...')
        log_file.write('Starting scheduling procedure for parcels from depots to households...\n')  
                    
        startValueProgress = 75.0 if doCrowdShipping else 60.0
        endValueProgress   = 80.0
        tourType = 0
        deliveries = create_schedules(parcelsUCC[0], dropOffTime, skimTravTime, skimDistance, parcelNodesCEP, parcelDepTime, 
                                      tourType, label, root, startValueProgress, endValueProgress) 
           
        # Depots to UCCs
        print('Starting scheduling procedure for parcels from depots to UCC...')
        log_file.write('Starting scheduling procedure for parcels from depots to UCC...\n')
        
        startValueProgress = 80.0
        endValueProgress   = 83.0
        tourType = 1
        deliveries1 = create_schedules(parcelsUCC[1], dropOffTime, skimTravTime, skimDistance, parcelNodesCEP, parcelDepTime, 
                                       tourType, label, root, startValueProgress, endValueProgress) 
                
    
        # Depots to UCCs (van)
        print('Starting scheduling procedure for parcels from UCCs (by van)...')
        log_file.write('Starting scheduling procedure for parcels from UCCs (by van)...\n')
        
        startValueProgress = 83.0
        endValueProgress   = 86.0     
        tourType = 2
        deliveries2 = create_schedules(parcelsUCC[2], dropOffTime, skimTravTime, skimDistance, parcelNodesCEP, parcelDepTime, 
                                       tourType, label, root, startValueProgress, endValueProgress) 
    
    
        # Depots to UCCs (LEVV)
        print('Starting scheduling procedure for parcels from UCCs (by LEVV)...')
        log_file.write('Starting scheduling procedure for parcels from UCCs (by LEVV)...\n')
        
        startValueProgress = 86.0
        endValueProgress   = 89.0
        tourType = 3            
        deliveries3 = create_schedules(parcelsUCC[3], dropOffTime, skimTravTime, skimDistance, parcelNodesCEP, parcelDepTime, 
                                       tourType, label, root, startValueProgress, endValueProgress) 
    
        
        # Combine deliveries of all tour types
        deliveries = pd.concat([deliveries, deliveries1, deliveries2, deliveries3])
        deliveries.index = np.arange(len(deliveries))
    
                
    # ----------- Scheduling of trips (REF scenario) ----------------------------
    
    if label != 'UCC':                 
        print('Starting scheduling procedure for parcels...'), log_file.write('Starting scheduling procedure for parcels...\n')    
        
        startValueProgress = 75.0 if doCrowdShipping else 60.0
        endValueProgress   = 90.0
        if varDict["Type"] == 'LastMile':
            tourType = 0
        if varDict["Type"] == 'Hub2Hub':
            tourType = 4
        if varDict["Type"] == 'CombinedConsol_LastMile':
            tourType = 0       
        
        deliveries = create_schedules(parcels, varDict["CAPACITY"], skimTravTime, skimDistance, parcelNodesCEP, parcelDepTime, 
                                      tourType, label, root, startValueProgress, endValueProgress)
    
        #Map vehicles!
        
        if varDict["Type"] == 'LastMile':
            deliveries ["VehType"] = deliveries["CEP"].map(varDict["VEHICLES"]).str[0] 
        if varDict["Type"] == 'Hub2Hub':
            deliveries ["VehType"] = deliveries["CEP"].map(varDict["VEHICLES"]).str[1]             
        if varDict["Type"] == 'CombinedConsol_LastMile':
            # deliveries ["VehType"] = deliveries["CEP"].map(varDict["VEHICLES"]).str[1]    
            deliveries["VehType"] = deliveries["CEP"].map(varDict["VEHICLES"])
            deliveries["VehType"] = np.where(deliveries["Type"] == "Delivery", deliveries["VehType"].str[0],np.where (deliveries["Type"] == "Pickup", deliveries["VehType"].str[0] , np.where ( deliveries["Type"] == "consolidated",deliveries["VehType"].str[1],"NA")))

    # ------------------ Export output table to CSV and SHP -------------------
    
    # Transform to MRDH zone numbers and export
    deliveries['O_zone']  =  [zoneDict[x] for x in deliveries['O_zone']]
    deliveries['D_zone']  =  [zoneDict[x] for x in deliveries['D_zone']]
    deliveries['TripDepTime'] = [round(deliveries['TripDepTime'][i], 3) for i in deliveries.index]
    deliveries['TripEndTime'] = [round(deliveries['TripEndTime'][i], 3) for i in deliveries.index]
    
    print(f"Writing scheduled trips to {datapathO}ParcelSchedule_{label}.csv")
    log_file.write(f"Writing scheduled trips to {datapathO}ParcelSchedule_{label}.csv\n")
    deliveries.to_csv(f"{datapathO}ParcelSchedule_{label}.csv", index=False)  
    
    if root != '':
        root.progressBar['value'] = 91.0
    
    '''EDIT 6-10: do not create GEOJSON'''            
    # print('Writing GeoJSON...'), log_file.write('Writing GeoJSON...\n')
    
    # # Initialize arrays with coordinates        
    # Ax = np.zeros(len(deliveries), dtype=int)
    # Ay = np.zeros(len(deliveries), dtype=int)
    # Bx = np.zeros(len(deliveries), dtype=int)
    # By = np.zeros(len(deliveries), dtype=int)
    
    # # Determine coordinates of LineString for each trip
    # tripIDs  = [x.split('_')[-1] for x in deliveries['Trip_ID']]
    # tourTypes = np.array(deliveries['TourType'], dtype=int)
    # depotIDs = np.array(deliveries['Depot_ID'])
    # for i in deliveries.index[:-1]:
    #     # First trip of tour
    #     if tripIDs[i] == '0' and tourTypes[i]<=1:
    #         Ax[i] = parcelNodes['X'][depotIDs[i]]
    #         Ay[i] = parcelNodes['Y'][depotIDs[i]]
    #         Bx[i] = zonesX[deliveries['D_zone'][i]]
    #         By[i] = zonesY[deliveries['D_zone'][i]]
    #     # Last trip of tour
    #     elif tripIDs[i+1] == '0' and tourTypes[i]<=1:
    #         Ax[i] = zonesX[deliveries['O_zone'][i]]
    #         Ay[i] = zonesY[deliveries['O_zone'][i]]                
    #         Bx[i] = parcelNodes['X'][depotIDs[i]]
    #         By[i] = parcelNodes['Y'][depotIDs[i]]
    #     # Intermediate trips of tour
    #     else:
    #         Ax[i] = zonesX[deliveries['O_zone'][i]]
    #         Ay[i] = zonesY[deliveries['O_zone'][i]]
    #         Bx[i] = zonesX[deliveries['D_zone'][i]]
    #         By[i] = zonesY[deliveries['D_zone'][i]]
    # # Last trip of last tour
    # i += 1
    # if tourTypes[i]<=1:
    #     Ax[i] = zonesX[deliveries['O_zone'][i]]
    #     Ay[i] = zonesY[deliveries['O_zone'][i]]                
    #     Bx[i] = parcelNodes['X'][depotIDs[i]]
    #     By[i] = parcelNodes['Y'][depotIDs[i]]
    # else:
    #     Ax[i] = zonesX[deliveries['O_zone'][i]]
    #     Ay[i] = zonesY[deliveries['O_zone'][i]]
    #     Bx[i] = zonesX[deliveries['D_zone'][i]]
    #     By[i] = zonesY[deliveries['D_zone'][i]]
            
    # Ax = np.array(Ax, dtype=str)
    # Ay = np.array(Ay, dtype=str)
    # Bx = np.array(Bx, dtype=str)
    # By = np.array(By, dtype=str)
    # nTrips = len(deliveries)
    
    # with open(datapathO + f"ParcelSchedule_{label}.geojson", 'w') as geoFile:
    #     geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
    #     for i in range(nTrips-1):
    #         outputStr = ""
    #         outputStr = outputStr + '{ "type": "Feature", "properties": '
    #         outputStr = outputStr + str(deliveries.loc[i,:].to_dict()).replace("'",'"')
    #         outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
    #         outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
    #         outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } },\n'
    #         geoFile.write(outputStr)
    #         if i%int(nTrips/10) == 0:
    #             print('\t' + str(int(round((i / nTrips)*100, 0))) + '%', end='\r')
    #             if root != '':
    #                 root.progressBar['value'] = 91.0 + (98.0 - 91.0) * (i / nTrips)
                
    #     # Bij de laatste feature moet er geen komma aan het einde
    #     i += 1
    #     outputStr = ""
    #     outputStr = outputStr + '{ "type": "Feature", "properties": '
    #     outputStr = outputStr + str(deliveries.loc[i,:].to_dict()).replace("'",'"')
    #     outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
    #     outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
    #     outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } }\n'
    #     geoFile.write(outputStr)
    #     geoFile.write(']\n')
    #     geoFile.write('}')
    
    # print(f'Parcel schedules written to {datapathO}ParcelSchedule_{label}.geojson'), log_file.write(f'Parcel schedules written to {datapathO}ParcelSchedule_{label}.geojson\n')        
    
    
    
    # ------------------------ Create and export trip matrices ----------------
    
    if exportTripMatrix:
        print('Generating trip matrix...'), log_file.write('Generating trip matrix...\n')
        cols = ['ORIG','DEST', 'N_TOT']
        deliveries['N_TOT'] = 1
        
        # Gebruik N_TOT om het aantal ritten per HB te bepalen, voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(deliveries, values=['N_TOT'], index=['O_zone','D_zone'], aggfunc=np.sum)
        pivotTable['ORIG'] = [x[0] for x in pivotTable.index] 
        pivotTable['DEST'] = [x[1] for x in pivotTable.index]
        pivotTable = pivotTable[cols]
        
        # Assume one intrazonal trip for each zone with multiple deliveries visited in a tour
        intrazonalTrips = {}
        for i in deliveries[deliveries['N_parcels']>1].index:
            zone = deliveries.at[i,'D_zone']
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
        
        pivotTable.to_csv(f"{datapathO}tripmatrix_parcels_{label}.txt", index=False, sep='\t')
        print(f'Trip matrix written to {datapathO}tripmatrix_parcels_{label}.txt'), log_file.write(f'Trip matrix written to {datapathO}tripmatrix_{label}.txt\n')
    
        deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24
        deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24
        
     
        # for tod in range(24):                
        #     # print(f'\t Also generating trip matrix for TOD {tod}...'), log_file.write(f'\t Also generating trip matrix for TOD {tod}...\n')
        #     output = deliveries[(deliveries['TripDepTime'] >= tod) & (deliveries['TripDepTime'] < tod+1)].copy()
        #     output['N_TOT'] = 1
            
        #     if len(output) > 0:
        #         # Gebruik deze dummies om het aantal ritten per HB te bepalen, voor elk logistiek segment, voertuigtype en totaal
        #         pivotTable = pd.pivot_table(output, values=['N_TOT'], index=['O_zone','D_zone'], aggfunc=np.sum)
        #         pivotTable['ORIG'] = [x[0] for x in pivotTable.index] 
        #         pivotTable['DEST'] = [x[1] for x in pivotTable.index]
        #         pivotTable = pivotTable[cols]
    
        #         # Assume one intrazonal trip for each zone with multiple deliveries visited in a tour
        #         intrazonalTrips = {}
        #         for i in output[output['N_parcels']>1].index:
        #             zone = output.at[i,'D_zone']
        #             if zone in intrazonalTrips.keys():
        #                 intrazonalTrips[zone] += 1
        #             else:
        #                 intrazonalTrips[zone] = 1           
        #         intrazonalKeys = list(intrazonalTrips.keys())
        #         for zone in intrazonalKeys:
        #             if (zone, zone) in pivotTable.index:
        #                 pivotTable.at[(zone, zone), 'N_TOT'] += intrazonalTrips[zone]
        #                 del intrazonalTrips[zone]            
        #         intrazonalTripsDF = pd.DataFrame(np.zeros((len(intrazonalTrips),3)), columns=cols)
        #         intrazonalTripsDF['ORIG' ] = intrazonalTrips.keys()
        #         intrazonalTripsDF['DEST' ] = intrazonalTrips.keys()
        #         intrazonalTripsDF['N_TOT'] = intrazonalTrips.values()
        #         pivotTable = pivotTable.append(intrazonalTripsDF)
        #         pivotTable = pivotTable.sort_values(['ORIG','DEST'])
        
        #     else:
        #         pivotTable = pd.DataFrame(columns=cols)
                
        #     pivotTable.to_csv(f"{datapathO}tripmatrix_parcels_{label}_TOD{tod}.txt", index=False, sep='\t')
            
    '''
    Here I add the consolidated tours with the hubhub parcels. It should replicate the procedure with the other parcels
    but with another vehicle size
    '''
        
    # --------------------------- End of module -------------------------------
    totaltime = round(time.time() - start_time, 2)
    log_file.write("Total runtime: %s seconds\n" % (totaltime))  
    log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
    log_file.close()    
    
    if root != '':
        root.update_statusbar("Parcel Scheduling: Done")
        root.progressBar['value'] = 100
        
        # 0 means no errors in execution
        root.returnInfo = [0, [0,0]]
        
        return root.returnInfo
    
    else:
        return [0, [0,0]]
        



#%% Function: create_schedules

def create_schedules(parcelsAgg, dropOffTimeDict, skimTravTime, skimDistance, parcelNodesCEP, parcelDepTime,
                     tourType, label, 
                     root, startValueProgress, endValueProgress):
    '''
    Create the parcel schedules and store them in a DataFrame
    '''
    nZones = int(len(skimTravTime)**0.5)
    depots = np.unique(parcelsAgg['Depot'])
    nDepots = len(depots)
    skimDist_flat = skimDistance.flatten()
    print('\t0%', end='\r')
    
    tours            = {}
    parcelsDelivered = {}
    departureTimes   = {}
    depotCount = 0
    nTrips     = 0
    
    for depot in np.unique(parcelsAgg['Depot']):
        depotParcels = parcelsAgg[parcelsAgg['Depot']==depot]
        
        vehicle = list(set((depotParcels["VEHTYPE"])))[0]
        
        
        dropOffTime = dropOffTimeDict [vehicle][1] / 3600 # Dropofftime to hours
        
        
        tours[depot]            = {}
        parcelsDelivered[depot] = {}
        departureTimes[depot]   = {}
        
        for cluster in np.unique(depotParcels['Cluster']):
            tour = []
            
            clusterParcels = depotParcels[depotParcels['Cluster']==cluster]
            depotZone = list(clusterParcels['Orig'])[0]
            destZones = list(clusterParcels['Dest'])
            nParcelsPerZone = dict(zip(destZones, clusterParcels['Parcels']))
            
            # Nearest neighbor
            tour.append(depotZone)
            for i in range(len(destZones)):
                distances = [skimDistance[tour[i] * nZones + dest] for dest in destZones]
                nextIndex = np.argmin(distances)
                tour.append(destZones[nextIndex])
                destZones.pop(nextIndex)
            tour.append(depotZone)
            
            # Shuffle the order of tour locations and accept the shuffle if it reduces the tour distance
            nStops = len(tour)
            tour = np.array(tour, dtype=int)
            tourDist = np.sum(skimDistance[tour[:-1] * nZones + tour[1:]])
            if nStops > 4:
                for shiftLocA in range(1, nStops-1):
                    for shiftLocB in range(1,nStops-1):
                        if shiftLocA != shiftLocB:
                            swappedTour             = tour.copy()
                            swappedTour[shiftLocA]  = tour[shiftLocB]
                            swappedTour[shiftLocB]  = tour[shiftLocA]
                            swappedTourDist = np.sum(skimDistance[swappedTour[:-1] * nZones + swappedTour[1:]])
                            
                            if swappedTourDist < tourDist:
                                tour = swappedTour.copy()
                                tourDist = swappedTourDist
            
            # Add current tour to dictionary with all formed tours             
            tours[depot][cluster] = list(tour.copy())
            
            # Store the number of parcels delivered at each location in the tour
            nParcelsPerStop = []
            for i in range(1, nStops-1):
                nParcelsPerStop.append(nParcelsPerZone[tour[i]])
            nParcelsPerStop.append(0)
            parcelsDelivered[depot][cluster] = list(nParcelsPerStop.copy())
            
            # Determine the departure time of each trip in the tour
            departureTimesTour = [np.where(parcelDepTime > np.random.rand())[0][0] + np.random.rand()]
            for i in range(1, nStops-1):
                orig = tour[i-1]
                dest = tour[i]
                travTime = skimTravTime[orig * nZones + dest] / 3600
                departureTimesTour.append(departureTimesTour[i-1] + 
                                          dropOffTime * nParcelsPerStop[i-1] +
                                          travTime)
            departureTimes[depot][cluster] = list(departureTimesTour.copy())
        
            nTrips += (nStops - 1)
            
        print('\t' + str(int(round(((depotCount+1)/nDepots)*100,0))) + '%', end='\r')
        
        if root != '':
            root.progressBar['value'] = startValueProgress + \
                                        (endValueProgress - startValueProgress - 1) * \
                                        (depotCount+1)/nDepots
    
        depotCount += 1
    
    
    # ------------------------------ Create return table ---------------------- 
    deliveriesCols = ['TourType',   'CEP',          'Depot_ID',     'Tour_ID', 
                      'Trip_ID',    'Unique_ID',    'O_zone',       'D_zone',       
                      'N_parcels', 'Traveltime', 'TourDepTime',  'TripDepTime', 
                      'TripEndTime', 'Type',"TourDist"]
    deliveries = np.zeros((nTrips, len(deliveriesCols)), dtype=object)

    tripcount = 0
    for depot in tours.keys():
        for tour in tours[depot].keys():
            for trip in range(len(tours[depot][tour])-1):
                orig = tours[depot][tour][trip]
                dest = tours[depot][tour][trip+1]
                deliveries[tripcount, 0] = tourType                                  # Depot to HH (0) or UCC (1), UCC to HH by van (2)/LEVV (3)
                if tourType <= 1:
                    deliveries[tripcount, 1] = parcelNodesCEP[depot]                 # Name of the couriers
                else:
                    # deliveries[tripcount, 1] = 'ConsolidatedUCC'                     # Name of the couriers  # Took it out because not sure why was this!
                    deliveries[tripcount, 1] = parcelNodesCEP[depot]
                if depot < 1000:
                    deliveries[tripcount, 2] = depot                                     # Depot_ID
                    deliveries[tripcount, 3] = f'{depot}_{tour}'                         # Tour_ID
                    deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'                  # Trip_ID
                    deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'       # Unique ID under consideration of tour type
                '''EDIT 6-10: SHADOW NODES - start'''            
                if depot >= 1000:
                    depot = depot - 1000
                    deliveries[tripcount, 2] = depot                                     # Depot_ID
                    deliveries[tripcount, 3] = f'{depot}_{tour}'                         # Tour_ID
                    deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'                  # Trip_ID
                    deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'       # Unique ID under consideration of tour type
                    depot = depot + 1000
                '''EDIT 6-10: SHADOW NODES - end'''            
                deliveries[tripcount, 6] = orig                                      # Origin
                deliveries[tripcount, 7] = dest                                      # Destination
                deliveries[tripcount, 8] = parcelsDelivered[depot][tour][trip]       # Number of parcels
                deliveries[tripcount, 9] = skimTravTime[orig * nZones + dest] / 3600 # Travel time in hrs
                deliveries[tripcount,10] = departureTimes[depot][tour][0]            # Departure of tour from depot
                deliveries[tripcount,11] = departureTimes[depot][tour][trip]         # Departure time of trip
                deliveries[tripcount,12] = 0.0                                       # End of trip/start of next trip if there is another one                
                '''EDIT 6-10: SHADOW NODES - start'''            
                deliveries[tripcount,13] = 'Delivery'                                                     
                if depot >= 1000: 
                    deliveries[tripcount,13] = 'Pickup'                              
                '''EDIT 6-10: SHADOW NODES - end'''
                deliveries[tripcount,14] = get_distance(orig, dest, skimDist_flat, nZones)
                # skimDistance[orig * nZones + dest]
                tripcount += 1
                
    # Place in DataFrame with the right data type per column
    deliveries = pd.DataFrame(deliveries, columns=deliveriesCols)
    dtypes =  {'TourType':int,      'CEP':str,          'Depot_ID':int,      'Tour_ID':str, 
               'Trip_ID':str,       'Unique_ID':str,    'O_zone':int,        'D_zone':int, 
               'N_parcels':int,     'Traveltime':float, 'TourDepTime':float, 'TripDepTime':float, 
               'TripEndTime':float, 'Type':str,"TourDist":float}
    for col in range(len(deliveriesCols)):
        deliveries[deliveriesCols[col]] = deliveries[deliveriesCols[col]].astype(dtypes[deliveriesCols[col]])

    vehTypes  = ['Van',   'Van',   'Van', 'LEVV','Consol']
    origTypes = ['Depot', 'Depot', 'UCC', 'UCC',"Depot"]
    destTypes = ['HH',    'UCC',   'HH',  'HH',"Depot"]

    deliveries['VehType' ] = vehTypes[tourType]
    deliveries['OrigType'] = origTypes[tourType]
    deliveries['DestType'] = destTypes[tourType]

    if root != '':
        root.progressBar['value'] = endValueProgress
                             
    return deliveries



#%% Function: cluster_parcels

def cluster_parcels(parcels, maxVehicleLoadDict, skimDistance,
                    root, startValueProgress, endValueProgress):
    '''
    Assign parcels to clusters based on spatial proximity with cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.
    '''
    parcels["NewIndex"] = parcels.index.tolist()
    depotNumbers = np.unique(parcels['DepotNumber'])
    nParcels = len(parcels)
    nParcelsAssigned = 0
    firstClusterID   = 0
    nZones = int(len(skimDistance)**0.5)
    
    parcels['Cluster'] = -1
    
    print('\t0%', end='\r')
    
    # First check for depot/destination combination with more than {maxVehicleLoad} parcels
    # These we don't need to use the clustering algorithm for
    # print (parcels)
    
    #TODO 
    #TO DO allow multiple vehicles?
    #parcels ['VEHTYPE'] = 'Van' # Making it such that there is one type of vans (TO CHANGE)
    
    ### SEPARATE DIFFERENT VEHICLE TYPES!
    
    
    # OrigParcels1 = parcels
    # parcels = OrigParcels1
    OrigParcels = parcels
    
    ClusterDict = {}
    
    for vehicletype in set(OrigParcels['VEHTYPE']):
        # print(vehicletype)
        maxVehicleLoad = maxVehicleLoadDict [vehicletype][0]
        parcels = OrigParcels[OrigParcels["VEHTYPE"] == vehicletype]
 
        
        counts = pd.pivot_table(parcels, values=['VEHTYPE'], index=['DepotNumber','D_zone'], aggfunc=len)
        
        
        
        
        whereLargeCluster = list(counts.index[np.where(counts>=maxVehicleLoad)[0]])
        for x in whereLargeCluster:
            depotNumber = x[0]
            destZone    = x[1]

            
            
            # indices = np.where((parcels['DepotNumber']==depotNumber) & (parcels['D_zone']==destZone))[0] # For some reason this brings indexes that are then nans. This brings an key error in next step
            
            
            largeCluster = parcels [(parcels['DepotNumber']==depotNumber) & (parcels['D_zone']==destZone)]
            indices = largeCluster.index.tolist()

            for i in range(int(np.floor(len(indices)/maxVehicleLoad))):

                parcels.loc[indices[:maxVehicleLoad], 'Cluster'] = firstClusterID
                indices = indices[maxVehicleLoad:]
                
                firstClusterID += 1
                nParcelsAssigned += maxVehicleLoad
                
                print('\t' + str(int(round((nParcelsAssigned/nParcels)*100,0))) + '%', end='\r')        
                if root != '':
                    root.progressBar['value'] = startValueProgress + \
                                                (endValueProgress - startValueProgress - 1) * \
                                                nParcelsAssigned/nParcels
                                                        
        # For each depot, cluster remaining parcels into batches of {maxVehicleLoad} parcels
        for depotNumber in depotNumbers:
            # Select parcels of the depot that are not assigned a cluster yet
            # print(depotNumber)
            parcelsToFit = parcels[(parcels['DepotNumber']==depotNumber) & 
                                   (parcels['Cluster']==-1)].copy()
            
            # Sort parcels descending based on distance to depot
            # so that at the end of the loop the remaining parcels are all nearby the depot
            # and form a somewhat reasonable parcels cluster
            parcelsToFit['Distance'] = skimDistance[(parcelsToFit['O_zone']-1) * nZones + 
                                                    (parcelsToFit['D_zone']-1)]
            parcelsToFit = parcelsToFit.sort_values('Distance', ascending=False)
            parcelsToFitIndex  = list(parcelsToFit.index)
            parcelsToFit.index = np.arange(len(parcelsToFit))
            dests  = np.array(parcelsToFit['D_zone'])
            
            # How many tours are needed to deliver these parcels
            nTours = int(np.ceil(len(parcelsToFit)/maxVehicleLoad))
            
            # In the case of 1 tour it's simple, all parcels belong to the same cluster
            if nTours == 1:
                parcels.loc[parcelsToFitIndex, 'Cluster'] = firstClusterID
                firstClusterID += 1
                nParcelsAssigned += len(parcelsToFit)
            
            # When there are multiple tours needed, the heuristic is a little bit more complex
            else:
                clusters = np.ones(len(parcelsToFit), dtype=int) * -1
                
                for tour in range(nTours):                
                    # Select the first parcel for the new cluster that is now initialized
                    yetAssigned    = (clusters!=-1)
                    notYetAssigned = np.where(~yetAssigned)[0]
                    firstParcelIndex = notYetAssigned[0]
                    clusters[firstParcelIndex] = firstClusterID
                    
                    # Find the nearest {maxVehicleLoad-1} parcels to this first parcel that are not in a cluster yet
                    distances = skimDistance[(dests[firstParcelIndex]-1) * nZones + (dests-1)]
                    distances[notYetAssigned[0]] = 99999
                    distances[yetAssigned]       = 99999
                    clusters[np.argsort(distances)[:(maxVehicleLoad-1)]] = firstClusterID
                                    
                    firstClusterID += 1
                
                # Group together remaining parcels, these are all nearby the depot
                yetAssigned    = (clusters!=-1)
                notYetAssigned = np.where(~yetAssigned)[0]
                clusters[notYetAssigned] = firstClusterID
                firstClusterID += 1
                    
                parcels.loc[parcelsToFitIndex, 'Cluster'] = clusters
                # parcels[parcels["NewIndex"].isin(parcelsToFitIndex)]['Cluster'] = clusters
                
                
                nParcelsAssigned += len(parcelsToFit)
    
                print('\t' + str(int(round((nParcelsAssigned/nParcels)*100,0))) + '%', end='\r')        
                if root != '':
                    root.progressBar['value'] = startValueProgress + \
                                                (endValueProgress - startValueProgress - 1) * \
                                                nParcelsAssigned/nParcels
                                                    
        
        
        
        parcels['Cluster'] = parcels['Cluster'].astype(int)
        p =parcels [['NewIndex','Cluster']]
        
        ClusterDict[vehicletype] = p
    
    p = pd.DataFrame(columns = ['NewIndex','Cluster'])
    for i,v in enumerate(ClusterDict):
        p = p.append (ClusterDict[v])
        
        
    p['NewIndex'] = np.int64(p['NewIndex'])     
    OrigParcels = OrigParcels.drop ('Cluster',axis=1)    
    parcels = pd.merge(OrigParcels,p,on='NewIndex', how='left')
    parcels = parcels.drop('NewIndex', axis=1)
    return parcels



#%% Function: do_crowdshipping



if not varDict['CONSOLIDATED_TRIPS'] :
    varDict['Type'] = 'LastMile'
    actually_run_module(args)
    
    #  Los unicos KPIs relevantes aca sonlos del lasst mile porque no hay consolidated flows
    
    parcels = pd.read_csv(varDict['Parcels']) 
    
    parcelschedule_hubspoke = pd.read_csv(f"{varDict['OUTPUTFOLDER']}ParcelSchedule_{varDict['LABEL']}.csv") # 
    parcelschedule_hubspoke_delivery = parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Delivery'].copy()
    parcelschedule_hubspoke_pickup = parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Pickup'].copy()
    
    
    parcelschedule_hubspoke_delivery['connected_tour'], parcelschedule_hubspoke_pickup['connected_tour'] = '', ''
    delivery_tour_end = parcelschedule_hubspoke_delivery.drop_duplicates(subset = ["Tour_ID"], keep='last').copy()
    pickup_tour_start = parcelschedule_hubspoke_pickup.drop_duplicates(subset = ["Tour_ID"], keep='first').copy()
    
    
    print(" ")
    print("Start KPIs")
    print(" ")
    
    KPIs['conventional_direct_return'] = 0
    if varDict['COMBINE_DELIVERY_PICKUP_TOUR']:   
        for index, delivery in delivery_tour_end.iterrows():
            endzone = delivery['O_zone']
            possible_pickuptours = pickup_tour_start[(
                (pickup_tour_start['CEP'] == delivery['CEP'])
                & (pickup_tour_start['O_zone'] == delivery['D_zone'])
                & (pickup_tour_start['connected_tour'] == '')
                & (pickup_tour_start['TourDepTime'] > delivery['TripDepTime'])
                )]
            possible_pickupzones = [invZoneDict[x]-1 for x in possible_pickuptours['D_zone']]
            if possible_pickupzones:
                best_tour = possible_pickuptours.iloc[skimTravTime[invZoneDict[endzone]-1,possible_pickupzones].argmin()]
                time_direct = skimTravTime[invZoneDict[delivery['O_zone']]-1, invZoneDict[best_tour['D_zone']]-1]
                time_via_depot = ( skimTravTime[invZoneDict[delivery['O_zone']]-1, invZoneDict[delivery['D_zone']]-1] +
                                    skimTravTime[invZoneDict[best_tour['O_zone']]-1, invZoneDict[best_tour['D_zone']]-1] )
                if time_direct < time_via_depot:
                    pickup_tour_start.loc[best_tour.name, 'connected_tour'] = delivery['Tour_ID']
                    parcelschedule_hubspoke_pickup.loc[best_tour.name, 'connected_tour'] = delivery['Tour_ID']
                    delivery_tour_end.loc[index, 'connected_tour'] = best_tour['Tour_ID']
                    parcelschedule_hubspoke_delivery.loc[index, 'connected_tour'] = best_tour['Tour_ID']
                    
                    KPIs['conventional_direct_return'] += round(get_distance(invZoneDict[delivery['O_zone']], invZoneDict[best_tour['D_zone']], skimDist_flat, nSkimZones),2)
                    # Shouldn't you be substracting the return distance to the depot?
                    
    '''Conventional KPI's'''
    # Parcel scheduling
    distances ={}
    distance = 0
    for cep in cepList:
        distances[str(cep)] = 0

    ParcelSchedules = [parcelschedule_hubspoke_delivery, parcelschedule_hubspoke_pickup]
    for ParcelSchedule in ParcelSchedules:
        for index, delivery in ParcelSchedule.iterrows():
            if delivery['connected_tour'] != '': continue
            else:
                orig = invZoneDict[delivery['O_zone']]
                dest = invZoneDict[delivery['D_zone']]
                distance += get_distance(orig, dest, skimDist_flat, nSkimZones)
                distances[delivery['CEP']] +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                

    LastMileDistance = round(distance, 2) 
    distances ["total"] = round(distance, 2) 
    for cep in cepList:
        distances[str(cep)] =  round(distances[str(cep)] , decimals=0) 
    
    
    
    KPIs['conventional_parcels_delivered'] = ParcelSchedules[0]['N_parcels'].sum()
    KPIs['conventional_parcels_picked-up'] = ParcelSchedules[1]['N_parcels'].sum() 
    KPIs['conventional_parcels_total'] = len(np.unique(parcels['Parcel_ID'].append(parcels['Parcel_ID'])))
    KPIs['conventional_distance'] = int(distance + KPIs['conventional_direct_return']) ### alleen den haag 
    KPIs['conventional_trips'] = sum([len(ParcelSchedule) for ParcelSchedule in ParcelSchedules])
    KPIs['conventional_tours'] = sum([len(np.unique(ParcelSchedule['Tour_ID'])) for ParcelSchedule in ParcelSchedules])
    KPIs['conventional_distance_avg'] = round(distance / KPIs['conventional_parcels_total'],2)
    
    CEPflowDeliver = {}
    CEPflowPuP = {}
    
    for courier,depots in cepNodeDict.items():
        CEPflowDeliver[courier] = {}
        CEPflowPuP[courier] ={} 
        Deliver = 0
        PickeUp = 0
        for depot in depots:
            #print("")
            Dep = depot +1 # In the dictionary it uses the Index, not the ID of the depot
            parcelsDelivered = ParcelSchedules[0][ParcelSchedules[0]['Depot_ID']==Dep]['N_parcels'].sum()
            parcelsPickedUp  = ParcelSchedules[1][ParcelSchedules[1]['Depot_ID']==Dep]['N_parcels'].sum()
            CEPflowDeliver[courier][Dep] = parcelsDelivered
            CEPflowPuP[courier][Dep] = parcelsPickedUp
            Deliver += parcelsDelivered
            PickeUp += parcelsPickedUp
        CEPflowDeliver[courier]['Sum'] = Deliver
        CEPflowPuP[courier]['Sum'] =PickeUp 
    
    KPIs["deliverFlows"] = CEPflowDeliver
    KPIs["pickUpFlows"] = CEPflowPuP
    KPIs["distances"] = distances

    vehicletypes = list((varDict["CAPACITY"]).keys())
    
    kmperVeh = {}
    
    for i in vehicletypes:
        kmperVeh[i] =0
    
    for i,v in enumerate(KPIs['LastMile distances']):
        if v != "total":
            kmperVeh[varDict["VEHICLES"][v][0]] += KPIs['LastMile distances'][v]
            


    for i,v in enumerate(kmperVeh):
        kmperVeh[v] = round (kmperVeh[v],2)


    KPIs["Kms per vehicle type"] = kmperVeh



else:

    # Como hay consolidated flows, hay que hacer de nuevo los tours
    
    if varDict['DEDICATED_CONSOLIDATED_TRIPS']:
    
        varDict['Type'] = 'LastMile'
        
        print(" ")
        print("Doing tours for last mile")
        print(" ")
        
        
        actually_run_module(args)
        
        
        print(" ")
        print("Doing tours for consolidated")
        print(" ")
        
    
        varDict['Type'] = 'Hub2Hub'
        
        
        actually_run_module(args)

    else: # Join the last mile and hub2hub in the same parcel file. The difference will be the time per parcel delay!!!
        
        varDict['Type'] = 'CombinedConsol_LastMile'
        
        
        actually_run_module(args)
        
        print("")


    print(" ")
    print("Start KPIs")
    print(" ")
    
    
    
    
    
    
    
    parcels = pd.read_csv(varDict['Parcels']) 
    
    if varDict['DEDICATED_CONSOLIDATED_TRIPS']:
        parcelschedule_hubspoke = pd.read_csv(f"{varDict['OUTPUTFOLDER']}ParcelSchedule_{varDict['LABEL']}_LastMile.csv") # 
        parcelschedule_hub2Hub = pd.read_csv(f"{varDict['OUTPUTFOLDER']}ParcelSchedule_{varDict['LABEL']}_Hub2Hub.csv") # 

    else:
        parcelschedule_hubspoke = pd.read_csv(f"{varDict['OUTPUTFOLDER']}ParcelSchedule_{varDict['LABEL']}_TotalUrbanDelivery.csv") # 
        
    parcelschedule_hubspoke_delivery = parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Delivery'].copy()
    parcelschedule_hubspoke_pickup = parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Pickup'].copy()

    
    parcelschedule_hubspoke_delivery['connected_tour'], parcelschedule_hubspoke_pickup['connected_tour'] = '', ''
    delivery_tour_end = parcelschedule_hubspoke_delivery.drop_duplicates(subset = ["Tour_ID"], keep='last').copy()
    pickup_tour_start = parcelschedule_hubspoke_pickup.drop_duplicates(subset = ["Tour_ID"], keep='first').copy()
    
    
    # print("por aca!!!")
    
    KPIs['conventional_direct_return'] = 0
    if varDict['COMBINE_DELIVERY_PICKUP_TOUR']:   
        for index, delivery in delivery_tour_end.iterrows():
            endzone = delivery['O_zone']
            possible_pickuptours = pickup_tour_start[(
                (pickup_tour_start['CEP'] == delivery['CEP'])
                & (pickup_tour_start['O_zone'] == delivery['D_zone'])
                & (pickup_tour_start['connected_tour'] == '')
                & (pickup_tour_start['TourDepTime'] > delivery['TripDepTime'])
                )]
            possible_pickupzones = [invZoneDict[x]-1 for x in possible_pickuptours['D_zone']]
            if possible_pickupzones:
                best_tour = possible_pickuptours.iloc[skimTravTime[invZoneDict[endzone]-1,possible_pickupzones].argmin()]
                time_direct = skimTravTime[invZoneDict[delivery['O_zone']]-1, invZoneDict[best_tour['D_zone']]-1]
                time_via_depot = ( skimTravTime[invZoneDict[delivery['O_zone']]-1, invZoneDict[delivery['D_zone']]-1] +
                                    skimTravTime[invZoneDict[best_tour['O_zone']]-1, invZoneDict[best_tour['D_zone']]-1] )
                if time_direct < time_via_depot:
                    pickup_tour_start.loc[best_tour.name, 'connected_tour'] = delivery['Tour_ID']
                    parcelschedule_hubspoke_pickup.loc[best_tour.name, 'connected_tour'] = delivery['Tour_ID']
                    delivery_tour_end.loc[index, 'connected_tour'] = best_tour['Tour_ID']
                    parcelschedule_hubspoke_delivery.loc[index, 'connected_tour'] = best_tour['Tour_ID']
                    
                    KPIs['conventional_direct_return'] += round(get_distance(invZoneDict[delivery['O_zone']], invZoneDict[best_tour['D_zone']], skimDist_flat, nSkimZones),2)
                    # Shouldn't you be substracting the return distance to the depot?
                    
    '''Conventional KPI's'''
    # Parcel scheduling
    
    distances ={}
    distance = 0
    for cep in cepList:
        distances[str(cep)] = 0

    ParcelSchedules = [parcelschedule_hubspoke_delivery, parcelschedule_hubspoke_pickup]
    for ParcelSchedule in ParcelSchedules:
        for index, delivery in ParcelSchedule.iterrows():
            if delivery['connected_tour'] != '': continue
            else:
                orig = invZoneDict[delivery['O_zone']]
                dest = invZoneDict[delivery['D_zone']]
                distance += get_distance(orig, dest, skimDist_flat, nSkimZones)
                distances[delivery['CEP']] +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                
                # if delivery['CEP'] == 'DHL': distanceDHL +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                # if delivery['CEP'] == 'DPD': distanceDPD +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                # if delivery['CEP'] == 'FedEx': distanceFedEx +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                # if delivery['CEP'] == 'GLS': distanceGLS +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                # if delivery['CEP'] == 'PostNL': distancePostNL +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                # if delivery['CEP'] == 'UPS': distanceUPS +=get_distance(orig, dest, skimDist_flat, nSkimZones)
    
    
    LastMileDistance = round(distance, 2) 
    distances ["total"] = round(distance, 2) 
    for cep in cepList:
        distances[str(cep)] =  round(distances[str(cep)] , 2) 
    
    
    KPIs['conventional_parcels_delivered'] = ParcelSchedules[0]['N_parcels'].sum()
    KPIs['conventional_parcels_picked-up'] = ParcelSchedules[1]['N_parcels'].sum() 
    KPIs['conventional_parcels_total'] = len(np.unique(parcels['Parcel_ID'].append(parcels['Parcel_ID'])))
    KPIs['conventional_distance'] = int(distance + KPIs['conventional_direct_return']) ### alleen den haag 
    KPIs['conventional_trips'] = sum([len(ParcelSchedule) for ParcelSchedule in ParcelSchedules])
    KPIs['conventional_tours'] = sum([len(np.unique(ParcelSchedule['Tour_ID'])) for ParcelSchedule in ParcelSchedules])
    KPIs['conventional_distance_avg'] = round(distance / KPIs['conventional_parcels_total'],2)
    
    CEPflowDeliver = {}
    CEPflowPuP = {}
    
    for courier,depots in cepNodeDict.items():
        CEPflowDeliver[courier] = {}
        CEPflowPuP[courier] ={} 
        Deliver = 0
        PickeUp = 0
        for depot in depots:
            #print("")
            Dep = depot +1 # In the dictionary it uses the Index, not the ID of the depot
            parcelsDelivered = ParcelSchedules[0][ParcelSchedules[0]['Depot_ID']==Dep]['N_parcels'].sum()
            parcelsPickedUp  = ParcelSchedules[1][ParcelSchedules[1]['Depot_ID']==Dep]['N_parcels'].sum()
            CEPflowDeliver[courier][int(Dep)] = int(parcelsDelivered)
            CEPflowPuP[courier][int(Dep)] = int(parcelsPickedUp)
            Deliver += parcelsDelivered
            PickeUp += parcelsPickedUp
        CEPflowDeliver[courier]['Sum'] = int(Deliver)
        CEPflowPuP[courier]['Sum'] = int(PickeUp )
    
    KPIs["deliverFlows"] = CEPflowDeliver
    KPIs["pickUpFlows"] = CEPflowPuP
    KPIs["LastMile distances"] = distances
    
    # print("por aca!!")
    # 
    #this is generating conosolidated KPI's. counting kilomters for each hub-hub trip made
    '''
    HERE ARETHE KPIs of the CONSOLIDATED
    
    parcels_hubhub = parcels_hubspoke.drop(['O_zone', 'D_zone', 'O_DepotNumber', 'D_DepotNumber', 'VEHTYPE'], axis=1)
    parcels_hubhub = parcels_hubhub[parcels_hubhub['Parcel_ID'].isin(parcels_hubspoke_Hague['Parcel_ID'])]
    
    parcels_hubhub = parcels_hubhub.drop(parcels_hubhub[parcels_hubhub['O_DepotZone'] == parcels_hubhub['D_DepotZone']].index)
    
    parcels_hyper_hubhub = parcel_trips[((parcel_trips['Network'] == 'conventional') & (parcel_trips['Type'] == 'consolidated'))]
    parcels_hyper_hubhub = parcels_hyper_hubhub.drop(['Network', 'Type'], axis=1)
    parcels_hyper_hubhub = parcels_hyper_hubhub.rename(columns={'O_zone': 'O_DepotZone', 'D_zone': 'D_DepotZone'}) #change the column names
    
    parcels_hubhub = parcels_hubhub.append(parcels_hyper_hubhub, ignore_index=True,sort=False)
    parcels_hubhub = parcels_hubhub.groupby(['O_DepotZone', 'D_DepotZone', 'CEP']).size().to_frame(name='Parcels').reset_index()
    for index, trip in parcels_hubhub.iterrows():
        parcels_hubhub.loc[index,'trips'] = np.ceil(trip['Parcels'] / varDict['PARCELS_MAXLOAD_Consolidated']) # TODO: if less than 180 parcels left, use a van
        parcels_hubhub.loc[index,'dist_single'] = get_distance(invZoneDict[trip['O_DepotZone']], invZoneDict[trip['D_DepotZone']], skimDist_flat, nSkimZones)
        parcels_hubhub.loc[index,'dist_total'] = parcels_hubhub.loc[index,'dist_single'] * parcels_hubhub.loc[index,'trips']
    KPIs['consolidated_distance'] = int(parcels_hubhub['dist_total'].sum())
    KPIs['consolidated_count'] = int(parcels_hubhub['Parcels'].sum())
    if KPIs['consolidated_count'] == 0: KPIs['consolidated_avg'] = 0
    else: KPIs['consolidated_avg'] = round(KPIs['consolidated_distance'] / KPIs['consolidated_count'],2)
    '''
    if varDict['DEDICATED_CONSOLIDATED_TRIPS']:    
        ConsolidatedPerCEP ={}
        Cons_distance = 0
        for cep in cepList:
            ConsolidatedPerCEP[str(cep)] = 0
        # ConsolidatedPerCEP ["ConsolidatedUCC"] =0
        # Cons_distance = 0
        # Cons_distanceDHL = 0
        # Cons_distanceDPD= 0
        # Cons_distanceFedEx= 0
        # Cons_distanceGLS= 0
        # Cons_distancePostNL = 0
        # Cons_distanceUPS= 0
        
        Condistance=0
        for index, delivery in parcelschedule_hub2Hub.iterrows():
            orig = invZoneDict[delivery['O_zone']]
            dest = invZoneDict[delivery['D_zone']]
            Condistance += get_distance(orig, dest, skimDist_flat, nSkimZones)
            ConsolidatedPerCEP[delivery['CEP']] +=get_distance(orig, dest, skimDist_flat, nSkimZones)
    
            Cons_distance +=Condistance
    
    
        
        ConsolidatedPerCEP ["total" ]= round(Condistance, 2) 
    
    
        for cep in cepList:
            ConsolidatedPerCEP[str(cep)] = round(ConsolidatedPerCEP[str(cep)] , 2) 
        
        KPIs['ConsolidatedPerCEP'] = ConsolidatedPerCEP
    else:  # Just 0 all the KPIs
        ConsolidatedPerCEP ={}
        Cons_distance = 0
        for cep in cepList:
            ConsolidatedPerCEP[str(cep)] = 0
        Condistance=0
        ConsolidatedPerCEP ["total" ]= round(Condistance, 2) 
        for cep in cepList:
            ConsolidatedPerCEP[str(cep)] = round(ConsolidatedPerCEP[str(cep)] , 2) 
        KPIs['ConsolidatedPerCEP'] = ConsolidatedPerCEP
        
        
    
    KPIs['TotalDistances']  = {
        
        "total" :round( Condistance + LastMileDistance,2)}
        
    for cep in cepList:
        if varDict['DEDICATED_CONSOLIDATED_TRIPS']:
            KPIs['TotalDistances'][cep] = round(ConsolidatedPerCEP[str(cep)] + distances[str(cep)],2)
        else:
            KPIs['TotalDistances'][cep] = round(0 + distances[str(cep)],2)
    # Add Kms per vehicle type
    
    vehicletypes = list((varDict["CAPACITY"]).keys())
    
    kmperVeh = {}
    
    for i in vehicletypes:
        kmperVeh[i] =0
    
    for i,v in enumerate(KPIs['LastMile distances']):
        if v != "total":
            kmperVeh[varDict["VEHICLES"][v][0]] += KPIs['LastMile distances'][v]
            



    for i,v in enumerate(KPIs['ConsolidatedPerCEP']):
        if v != "total":
            kmperVeh[varDict["VEHICLES"][v][1]] += KPIs['LastMile distances'][v]    

    for i,v in enumerate(kmperVeh):
        kmperVeh[v] = round (kmperVeh[v],2)


    KPIs["Kms per vehicle type"] = kmperVeh





    

         # This is missing some consolidated trips
    
KPIfile = varDict['OUTPUTFOLDER'] + 'KPI_' + varDict['LABEL']+'.json'

# Write KPIs as Json


# For some reason, json doesn't like np.int or floats
for index, key in enumerate(KPIs):
    # print(key)
    if type(KPIs[key]) == 'dict':
        for i,k in enumerate (key):
            print(k)
            if type(key[k]) == 'dict':
                for j,l in enumerate(k):
                    try:
                        val = k[l].item() 
                        k[l] = val
                        key[k] = k
                    except:
                        a=1
            else:
                try:
                    val = key[k].item() 
                    key[k] = val
                    KPIs[key] = key
                except:
                    a=1
    else:
        try:
            val = KPIs[key].item()  
            KPIs[key] = val
        except:
            a=1
# print(KPIs)


f = open(KPIfile, "w")
json.dump(KPIs, f,indent = 2)
f.close()


KPI_Json = json.dumps(KPIs, indent = 2) 
if varDict['printKPI'] :
    print(KPI_Json)

    
    
    
    

    
    
    

    
    

















