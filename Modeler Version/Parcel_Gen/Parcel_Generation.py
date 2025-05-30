# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:36:51 2021

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
            
        if sys.argv[0] == '' :
            params_file = open(f'{datapath}/Input/Params_ParcelGen.txt')
            varDict['LABEL'	]			= 'ParcelLockers'			
            varDict['DATAPATH']			= f'{datapath}'							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/' +'Input' +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/' + 'Output' +'/'			
            
            varDict['SKIMTIME'] 		= varDict['INPUTFOLDER'] + 'skimTijd_new_REF.mtx'  #'skimTijd_new_REF.mtx' 		
            varDict['SKIMDISTANCE']		= varDict['INPUTFOLDER'] + 'skimAfstand_new_REF.mtx' #'skimAfstand_new_REF.mtx'	
            varDict['ZONES']			= varDict['INPUTFOLDER'] +'Zones_v4.shp'#'Zones_v4.shp'				
            varDict['SEGS']				= varDict['INPUTFOLDER'] + 'SEGS2020.csv'	 #'SEGS2020.csv'				
            varDict['PARCELNODES']		= varDict['INPUTFOLDER'] + 'parcelNodes_v2Cycloon.shp' #'parcelNodes_v2.shp'				
            varDict['CEP_SHARES']       = varDict['INPUTFOLDER'] + 'CEPsharesCycloon.csv'	 #'CEPshares.csv'	
            varDict['ExternalZones']    = varDict['INPUTFOLDER'] + 'SupCoordinatesID.csv' #'CEPshares.csv'		

            
        else:  # This is the part for line cod execution
            locationparam = f'{datapath}' +'/'+ sys.argv[2] +'/' + sys.argv[4]
            params_file = open(locationparam)
            varDict['LABEL'	]			= sys.argv[1]				
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/' + sys.argv[2] +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/' + sys.argv[3] +'/'			
            
            varDict['SKIMTIME'] 		= varDict['INPUTFOLDER'] + sys.argv[5] #'skimTijd_new_REF.mtx' 		
            varDict['SKIMDISTANCE']		= varDict['INPUTFOLDER'] + sys.argv[6] #'skimAfstand_new_REF.mtx'	
            varDict['ZONES']			= varDict['INPUTFOLDER'] + sys.argv[7] #'Zones_v4.shp'				
            varDict['SEGS']				= varDict['INPUTFOLDER'] + sys.argv[8] #'SEGS2020.csv'				
            varDict['PARCELNODES']		= varDict['INPUTFOLDER'] + sys.argv[9] #'parcelNodes_v2.shp'				
            varDict['CEP_SHARES']       = varDict['INPUTFOLDER'] + sys.argv[10] #'CEPshares.csv'	
            varDict['ExternalZones']       = varDict['INPUTFOLDER'] + sys.argv[11] #'CEPshares.csv'			

            
            
        for line in params_file:
            if len(line.split('=')) > 1:
                key, value = line.split('=')
                if len(value.split(';')) > 1:
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
        varDict['RUN_DEMAND_MODULE']            = True
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
        
       
        # Hague
        varDict['Gemeenten_CS']         = ["sGravenhage", "Zoetermeer", "Midden_Delfland"]
        varDict['SCORE_ALPHAS']         = [0, 0, 0.1, 1]
        varDict['SCORE_COSTS']          = [0.2, .02, .02, 0,0] # tour_based, consolidated, hub, cs_trans, #interCEP_cost
        varDict['CONSOLIDATED_MAXLOAD'] = 500
        
        '''FOR PARCEL DEMAND MODULE'''
        # Changed parameters to C2X ACM post&pakketmonitor2020 20.8M parcels 
        varDict['PARCELS_PER_HH_C2C']   = 20.8 / 250 / 8.0 # M parcels / days / M HHs 
        varDict['PARCELS_PER_HH_B2C']   = 0.195
        varDict['PARCELS_PER_HH']       = varDict['PARCELS_PER_HH_C2C'] + varDict['PARCELS_PER_HH_B2C']
        varDict['PARCELS_PER_EMPL']     = 0
        varDict['Local2Local']          = 0.04
        varDict['CS_cust_willingness']  = 0.05 # Willingess to SEND a parcel by CS
        varDict['PARCELS_MAXLOAD']	    = 180												
        varDict['PARCELS_DROPTIME' ]    = 120												
        varDict['PARCELS_SUCCESS_B2C']   = 0.75											
        varDict['PARCELS_SUCCESS_B2B' ]  = 0.95											
        varDict['PARCELS_GROWTHFREIGHT'] = 1.0										

    
    args = ['', varDict]
    return args, varDict

method = 'from_file' #either from_file or from_code


args, varDict = generate_args(method)

TESTRUN = False # True to fasten further code TEST (runs with less parcels)
TestRunLen = 100










#%%


#%% Module 0: Load input data
'''
These variables will be used throughout the whole model
'''

Comienzo = dt.datetime.now()
# print ("Comienzo: ",Comienzo)
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


# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:20:50 2020

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
        self.root.title("Progress Parcel Demand")
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
        

def parcelsSelector(parcels, arr):
    parcel_lockers = list()  # check!!
    for i in range(len(arr)):
        parcel_lockers[i] = parcels[parcels['D_zone']==arr[i]]
    return(parcel_lockers)


def actually_run_module(args):

    try:
        # -------------------- Define datapaths -----------------------------------
        
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
        cepSharesPath    = varDict['CEP_SHARES']
        segsPath         = varDict['SEGS']    
        label            = varDict['LABEL']
        
        parcelsPerHH     = varDict['PARCELS_PER_HH']
        parcelsPerEmpl   = varDict['PARCELS_PER_EMPL']
        parcelSuccessB2B = varDict['PARCELS_SUCCESS_B2B']
        parcelSuccessB2C = varDict['PARCELS_SUCCESS_B2C']

        log_file = open(datapathO + "Logfile_ParcelDemand.log", "w")
        log_file.write("Start simulation at: " + datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
         
            
        # ---------------------------- Import data --------------------------------
        print('Importing data...'), log_file.write('Importing data...\n')
        zones = read_shape(zonesPath)
        zones = pd.DataFrame(zones).sort_values('AREANR')
        zones.index = zones['AREANR']
        supCoordinates = pd.read_csv(varDict['ExternalZones'], sep=',')
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
                
        segs              = pd.read_csv(segsPath)
        segs.index        = segs['zone']
        
        parcelNodes, coords = read_shape(parcelNodesPath, returnGeometry=True)
        parcelNodes['X']    = [coords[i]['coordinates'][0] for i in range(len(coords))]
        parcelNodes['Y']    = [coords[i]['coordinates'][1] for i in range(len(coords))]
        parcelNodes.index   = parcelNodes['id'].astype(int)
        parcelNodes         = parcelNodes.sort_index()    
        nParcelNodes        = len(parcelNodes)
           
        cepShares = pd.read_csv(cepSharesPath, index_col=0)
        cepList   = np.unique(parcelNodes['CEP'])
        cepNodes = [np.where(parcelNodes['CEP']==str(cep))[0] for cep in cepList]
        cepNodeDict = {}
        for cepNo in range(len(cepList)):
            cepNodeDict[cepList[cepNo]] = cepNodes[cepNo]
        
        
        # ------------------ Get skim data and make parcel skim --------------------
        skimTravTime = read_mtx(skimTravTimePath)
        nZones   = int(len(skimTravTime)**0.5)
        parcelSkim = np.zeros((nZones, nParcelNodes))
            
        # Skim with travel times between parcel nodes and all other zones
        i = 0
        for parcelNodeZone in parcelNodes['AREANR']:
            orig = invZoneDict[parcelNodeZone]
            dest = 1 + np.arange(nZones)
            parcelSkim[:,i] = np.round( (skimTravTime[(orig-1)*nZones+(dest-1)] / 3600),4)     
            i += 1
        
        
        # ---- Generate parcels each zone based on households and select a parcel node for each parcel -----
        print('Generating parcels...'), log_file.write('Generating parcels...\n')
        
        # Filter the zones of the study area (edit 24/4)
        
        
        zones = zones[zones['GEMEENTEN'].isin(varDict['Gemeenten_studyarea'])]
        
        
        zones['ParcelLockerAdoption'] = zones['AREANR'].isin(varDict['parcelLockers_zones']) * varDict['PL_ZonalDemand']
        zones['parcelSuccessB2C'] = zones['ParcelLockerAdoption'] + (1-zones['ParcelLockerAdoption'])*parcelSuccessB2C   #  This makes the parcel locker adoption affect the overall success rate	
        zones['parcelSuccessB2B'] = zones['ParcelLockerAdoption'] + (1-zones['ParcelLockerAdoption'])*parcelSuccessB2B  



        
        # Calculate number of parcels per zone based on number of households and total number of parcels on an average day
        zones['woningen'] = segs['1: woningen'        ] 
        zones['arbeidspl_totaal'] =segs['9: arbeidspl_totaal']
        
        zones['parcels'] = zones['woningen'] * parcelsPerHH  / zones['parcelSuccessB2C']
        zones['parcels'] +=  zones['arbeidspl_totaal'] * parcelsPerEmpl / zones['parcelSuccessB2B']
        
        # zones['parcels']  = (segs['1: woningen'        ] * parcelsPerHH   / parcelSuccessB2C)
        # zones['parcels'] += (segs['9: arbeidspl_totaal'] * parcelsPerEmpl / parcelSuccessB2B)
        
        
        zones['parcels']  = np.array(np.round(zones['parcels'],0), dtype=int)
        
        
        ParcelDemand =  np.round( sum(zones['woningen'] * parcelsPerHH + zones['arbeidspl_totaal'] * parcelsPerEmpl ) ,0) # Demand without counting success rate
    
    
    
        # zones['ParcelLockerAdoption'] = zones['AREANR'].isin(varDict['parcelLockers_zones']) * varDict['PL_ZonalDemand']
        # zones['parcelSuccessB2C'] = zones['ParcelLockerAdoption'] + (1-zones['ParcelLockerAdoption'])*parcelSuccessB2C   	
        # zones['parcelSuccessB2B'] = zones['ParcelLockerAdoption'] + (1-zones['ParcelLockerAdoption'])*parcelSuccessB2B          
        
        
        
        # Spread over couriers based on market shares
        for cep in cepList:
            zones['parcels_' + str(cep)] = np.round(cepShares['ShareTotal'][cep] * zones['parcels'], 0)
            zones['parcels_' + str(cep)] = zones['parcels_' + str(cep)].astype(int)
        
        # Total number of parcels per courier
        nParcels  = int(zones[["parcels_"+str(cep) for cep in cepList]].sum().sum())
        
        # Put parcel demand in Numpy array (faster indexing)
        cols    = ['Parcel_ID', 'O_zone', 'D_zone', 'DepotNumber']
        parcels = np.zeros((nParcels,len(cols)), dtype=int)
        parcelsCep = np.array(['' for i in range(nParcels)], dtype=object)
        
        # Now determine for each zone and courier from which depot the parcels are delivered
        count = 0
        for zoneID in zones['AREANR'] : 
            
            if zones['parcels'][zoneID] > 0: # Go to next zone if no parcels are delivered here
            
                for cep in cepList:
                    # Select dc based on min in parcelSkim
                    parcelNodeIndex = cepNodeDict[cep][parcelSkim[invZoneDict[zoneID]-1,cepNodeDict[cep]].argmin()]
                
                    # Fill allParcels with parcels, zone after zone. Parcels consist of ID, D and O zone and parcel node number
                    # in ongoing df from index count-1 the next x=no. of parcels rows, fill the cell in the column Parcel_ID with a number 
                    n = zones.loc[zoneID,'parcels_' + str(cep)]
                    parcels[count:count+n,0]  = np.arange(count+1, count+1+n,dtype=int)                    
                    parcels[count:count+n,1]  = parcelNodes['AREANR'][parcelNodeIndex+1]
                    parcels[count:count+n,2]  = zoneID
                    parcels[count:count+n,3]  = parcelNodeIndex + 1
                    parcelsCep[count:count+n] = cep
                
                    count += zones['parcels_' + str(cep)][zoneID]
        
        # Put the parcel demand data back in a DataFrame
        parcels = pd.DataFrame(parcels, columns=cols)
        parcels['CEP'] = parcelsCep
        
        
        # Make PL demand
        lockers_arr = varDict['parcelLockers_zones']
        parcels['PL']=0 #creating a column filled with 0
        PL_ZonalDemand = varDict['PL_ZonalDemand']
        temp = parcels[parcels['D_zone'].isin(lockers_arr)].sample(frac = PL_ZonalDemand) #creating a temporary dataframe filled with parcels, selected among the ones with a D_zone in which there is a PL (chosing randomly through PL_ZonalDemand since not every parcel will be delivered in that way)
        parcelsID = temp.Parcel_ID.unique()
        parcels.loc[parcels['Parcel_ID'].isin(parcelsID),'PL'] = parcels['D_zone'] # Here we are allowing that only the parcels of a zone are capable of using lockers. This can be changed for neighbouring zones and this line should be updated

        
        
        
        
        
        
        # Default vehicle type for parcel deliveries: vans
        parcels['VEHTYPE'] = 7

        # Rerouting through UCCs in the UCC-scenario
        if label == 'UCC': 
            
            vtNamesUCC = ['LEVV','Moped','Van','Truck','TractorTrailer','WasteCollection','SpecialConstruction']
            nLogSeg = 8
            
            # Logistic segment is 6: parcels
            ls = 6

            # Write the REF parcel demand
            print(f"Writing parcels to {datapathO}ParcelDemand_REF.csv"), log_file.write(f"Writing parcels to {datapathO}ParcelDemand_REF.csv\n")
            parcels.to_csv(f"{datapathO}ParcelDemand_REF.csv", index=False)  

            # Consolidation potential per logistic segment (for UCC scenario)
            probConsolidation = np.array(pd.read_csv(datapathI + 'ConsolidationPotential.csv', index_col='Segment'))
            
            # Vehicle/combustion shares (for UCC scenario)
            sharesUCC  = pd.read_csv(datapathI + 'ZEZscenario.csv', index_col='Segment')
        
            # Assume no consolidation potential and vehicle type switch for dangerous goods
            sharesUCC = np.array(sharesUCC)[:-1,:-1]
            
            # Only vehicle shares (summed up combustion types)
            sharesVehUCC = np.zeros((nLogSeg-1,len(vtNamesUCC)))
            for ls in range(nLogSeg-1):
                sharesVehUCC[ls,0] = np.sum(sharesUCC[ls,0:5])
                sharesVehUCC[ls,1] = np.sum(sharesUCC[ls,5:10])
                sharesVehUCC[ls,2] = np.sum(sharesUCC[ls,10:15])
                sharesVehUCC[ls,3] = np.sum(sharesUCC[ls,15:20])
                sharesVehUCC[ls,4] = np.sum(sharesUCC[ls,20:25])
                sharesVehUCC[ls,5] = np.sum(sharesUCC[ls,25:30])            
                sharesVehUCC[ls,6] = np.sum(sharesUCC[ls,30:35])
                sharesVehUCC[ls,:] = np.cumsum(sharesVehUCC[ls,:]) / np.sum(sharesVehUCC[ls,:])

            # Couple these vehicle types to Harmony vehicle types
            vehUccToVeh = {0:8, 1:9, 2:7, 3:1, 4:5, 5:6, 6:6}                

            print('Redirecting parcels via UCC...'), log_file.write('Redirecting parcels via UCC...\n')
            
            parcels['FROM_UCC'] = 0
            parcels['TO_UCC'  ] = 0
          
            destZones    = np.array(parcels['D_zone'].astype(int))
            depotNumbers = np.array(parcels['DepotNumber'].astype(int))
            whereDestZEZ = np.where((zones['ZEZ'][destZones]==1) & (probConsolidation[ls][0] > np.random.rand(len(parcels))))[0]
                        
            newParcels = np.zeros(parcels.shape, dtype=object)

            count = 0
            
            for i in whereDestZEZ:
                                  
                trueDest = destZones[i]
                
                # Redirect to UCC
                parcels.at[i,'D_zone'] = zones['UCC_zone'][trueDest]
                parcels.at[i,'TO_UCC'] = 1
                
                # Add parcel set to ZEZ from UCC
                newParcels[count, 1] = zones['UCC_zone'][trueDest]  # Origin
                newParcels[count, 2] = trueDest             # Destination
                newParcels[count, 3] = depotNumbers[i]      # Depot ID
                newParcels[count, 4] = parcelsCep[i]        # Courier name
                newParcels[count, 5] = vehUccToVeh[np.where(sharesVehUCC[ls,:]>np.random.rand())[0][0]] # Vehicle type
                newParcels[count, 6] = 1                    # From UCC
                newParcels[count, 7] = 0                    # To UCC
                
                count += 1

            newParcels = pd.DataFrame(newParcels)
            newParcels.columns = parcels.columns            
            newParcels = newParcels.iloc[np.arange(count),:]
            
            dtypes = {'Parcel_ID':int, 'O_zone':int,  'D_zone':int,   'DepotNumber':int, \
                      'CEP':str,       'VEHTYPE':int, 'FROM_UCC':int, 'TO_UCC':int}
            for col in dtypes.keys():
                newParcels[col] = newParcels[col].astype(dtypes[col])
            
            parcels = parcels.append(newParcels)        
            parcels.index = np.arange(len(parcels))
            parcels['Parcel_ID'] = np.arange(1,len(parcels)+1)
            
            nParcels = len(parcels)
            
            
        # ------------------------- Prepare output -------------------------------- 
        print(f"Writing parcels CSV to     {datapathO}ParcelDemand_{label}.csv"), log_file.write(f"Writing parcels to {datapathO}ParcelDemand_{label}.csv\n")
        parcels.to_csv(f"{datapathO}ParcelDemand_{label}.csv", index=False)  
         
        
        
        KPIs ["Number Of Parcels to be delivered per day (with redelivery)"] = len(parcels)
        
        KPIs ["New parcel daily demand"] = ParcelDemand
        
        KPIs ["Redelivered parcels"] =  len(parcels) - ParcelDemand
        
        KPIs ["Number Of Parcels to Parcel lockers"] = np.count_nonzero(parcels['PL'])
        
        for cep in cepList:
            parcelCEP =parcels[parcels["CEP"] == cep]
            KPIs['ParcelDemand_' + cep] =  len(parcelCEP)
            
        
        
        # # Aggregate to number of parcels per zone and export to geojson
        # print(f"Writing parcels GeoJSON to {datapathO}ParcelDemand_{label}.geojson"), log_file.write(f"Writing shapefile to {datapathO}ParcelDemand_{label}.geojson\n")
        # if label == 'UCC':     
        #     parcelsShape = pd.pivot_table(parcels, values=['Parcel_ID'], index=["DepotNumber", 'CEP','D_zone', 'O_zone', 'VEHTYPE', 'FROM_UCC', 'TO_UCC'],\
        #                                   aggfunc = {'DepotNumber': np.mean, 'CEP':     'first',  'O_zone': np.mean, 'D_zone': np.mean, 'Parcel_ID': 'count', \
        #                                              'VEHTYPE':     np.mean, 'FROM_UCC': np.mean, 'TO_UCC': np.mean})
        #     parcelsShape = parcelsShape.rename(columns={'Parcel_ID':'Parcels'})
        #     parcelsShape = parcelsShape.set_index(np.arange(len(parcelsShape)))
        #     parcelsShape = parcelsShape.reindex(columns=[ 'O_zone','D_zone', 'Parcels', 'DepotNumber', 'CEP','VEHTYPE', 'FROM_UCC', 'TO_UCC'])
        #     parcelsShape = parcelsShape.astype({'DepotNumber': int, 'O_zone': int, 'D_zone': int, 'Parcels': int, 'VEHTYPE': int, 'FROM_UCC': int, 'TO_UCC': int})
        
        # else:
        #     parcelsShape = pd.pivot_table(parcels, values=['Parcel_ID'], index=["DepotNumber", 'CEP', 'D_zone', 'O_zone'],\
        #                                   aggfunc = {'DepotNumber': np.mean, 'CEP':'first', 'O_zone': np.mean, 'D_zone': np.mean, 'Parcel_ID': 'count'})
        #     parcelsShape = parcelsShape.rename(columns={'Parcel_ID':'Parcels'})
        #     parcelsShape = parcelsShape.set_index(np.arange(len(parcelsShape)))
        #     parcelsShape = parcelsShape.reindex(columns=[ 'O_zone','D_zone', 'Parcels', 'DepotNumber', 'CEP'])
        #     parcelsShape = parcelsShape.astype({'DepotNumber': int, 'O_zone': int, 'D_zone': int, 'Parcels': int})


        # # Initialize arrays with coordinates        
        # Ax = np.zeros(len(parcelsShape), dtype=int)
        # Ay = np.zeros(len(parcelsShape), dtype=int)
        # Bx = np.zeros(len(parcelsShape), dtype=int)
        # By = np.zeros(len(parcelsShape), dtype=int)
        
        # # Determine coordinates of LineString for each trip
        # depotIDs = np.array(parcelsShape['DepotNumber'])
        # for i in parcelsShape.index:
        #     if label == 'UCC' and parcelsShape.at[i, 'FROM_UCC'] == 1:
        #             Ax[i] = zonesX[parcelsShape['O_zone'][i]]
        #             Ay[i] = zonesY[parcelsShape['O_zone'][i]]
        #             Bx[i] = zonesX[parcelsShape['D_zone'][i]]
        #             By[i] = zonesY[parcelsShape['D_zone'][i]]
        #     else:
        #         Ax[i] = parcelNodes['X'][depotIDs[i]]
        #         Ay[i] = parcelNodes['Y'][depotIDs[i]]
        #         Bx[i] = zonesX[parcelsShape['D_zone'][i]]
        #         By[i] = zonesY[parcelsShape['D_zone'][i]]
                
        # Ax = np.array(Ax, dtype=str)
        # Ay = np.array(Ay, dtype=str)
        # Bx = np.array(Bx, dtype=str)
        # By = np.array(By, dtype=str)
        # nRecords = len(parcelsShape)
        
        # with open(datapathO + f"ParcelDemand_{label}.geojson", 'w') as geoFile:
        #     geoFile.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
        #     for i in range(nRecords-1):
        #         outputStr = ""
        #         outputStr = outputStr + '{ "type": "Feature", "properties": '
        #         outputStr = outputStr + str(parcelsShape.loc[i,:].to_dict()).replace("'",'"')
        #         outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
        #         outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
        #         outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } },\n'
        #         geoFile.write(outputStr)
        #         if i%int(nRecords/10) == 0:
        #             print('\t' + str(int(round((i / nRecords)*100, 0))) + '%', end='\r')
                    
        #     # Bij de laatste feature moet er geen komma aan het einde
        #     i += 1
        #     outputStr = ""
        #     outputStr = outputStr + '{ "type": "Feature", "properties": '
        #     outputStr = outputStr + str(parcelsShape.loc[i,:].to_dict()).replace("'",'"')
        #     outputStr = outputStr + ', "geometry": { "type": "LineString", "coordinates": [ [ '
        #     outputStr = outputStr + Ax[i] + ', ' + Ay[i] + ' ], [ '
        #     outputStr = outputStr + Bx[i] + ', ' + By[i] + ' ] ] } }\n'
        #     geoFile.write(outputStr)
        #     geoFile.write(']\n')
        #     geoFile.write('}')
        
        KPIfile = varDict['OUTPUTFOLDER'] + 'KPI_' + varDict['LABEL']+'.json'
    
    # Write KPIs as Json
    
        print('Gathering KPIs')
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

    
        totaltime = round(time.time() - start_time, 2)
        log_file.write("Total runtime: %s seconds\n" % (totaltime))  
        log_file.write("End simulation at: "+datetime.datetime.now().strftime("%y-%m-%d %H:%M")+"\n")
        log_file.close()    
        
        if root != '':
            root.update_statusbar("Parcel Demand: Done")
            root.progressBar['value'] = 100
            
            # 0 means no errors in execution
            root.returnInfo = [0, [0,0]]
            
            return root.returnInfo
        
        else:
            return [0, [0,0]]
            
        
    except BaseException:
        import sys
        log_file.write(str(sys.exc_info()[0])), log_file.write("\n")
        import traceback
        log_file.write(str(traceback.format_exc())), log_file.write("\n")
        log_file.write("Execution failed!")
        log_file.close()
        
        if root != '':
            # Use this information to display as error message in GUI
            root.returnInfo = [1, [sys.exc_info()[0], traceback.format_exc()]]
            
            if __name__ == '__main__':
                root.update_statusbar("Parcel Demand: Execution failed!")
                errorMessage = 'Execution failed!\n\n' + str(root.returnInfo[1][0]) + '\n\n' + str(root.returnInfo[1][1])
                root.error_screen(text=errorMessage, size=[900,350])                
            
            else:
                return root.returnInfo
        else:
            return [1, [sys.exc_info()[0], traceback.format_exc()]]
 
    
    
#%% For if you want to run the module from this script itself (instead of calling it from the GUI module)
    
    
# else:

#     if __name__ == '__main__':
        
#         INPUTFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/2016/'
#         OUTPUTFOLDER = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/output/RunREF2016/'
#         PARAMFOLDER	 = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/parameters/'
        
#         SKIMTIME        = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimTijd_REF.mtx'
#         SKIMDISTANCE    = 'P:/Projects_Active/18007 EC HARMONY/Work/WP6/MassGT_v11/data/LOS/2016/skimAfstand_REF.mtx'
#         LINKS		    = INPUTFOLDER + 'links_v5.shp'
#         NODES           = INPUTFOLDER + 'nodes_v5.shp'
#         ZONES           = INPUTFOLDER + 'Zones_v4.shp'
#         SEGS            = INPUTFOLDER + 'SEGS2020.csv'
#         COMMODITYMATRIX = INPUTFOLDER + 'CommodityMatrixNUTS2_2016.csv'
#         PARCELNODES     = INPUTFOLDER + 'parcelNodes_v2.shp'
        
#         YEARFACTOR = 193
        
#         NUTSLEVEL_INPUT = 2
        
#         PARCELS_PER_HH	 = 0.195
#         PARCELS_PER_EMPL = 0.073
#         PARCELS_MAXLOAD	 = 180
#         PARCELS_DROPTIME = 120
#         PARCELS_SUCCESS_B2C   = 0.75
#         PARCELS_SUCCESS_B2B   = 0.95
#         PARCELS_GROWTHFREIGHT = 1.0
        
#         SHIPMENTS_REF = ""
#         SELECTED_LINKS = ""
        
#         IMPEDANCE_SPEED = 'V_FR_OS'
        
#         LABEL = 'REF'
        
#         MODULES = ['SIF', 'SHIP', 'TOUR','PARCEL_DMND','PARCEL_SCHD','TRAF','OUTP']
        
#         args = [INPUTFOLDER, OUTPUTFOLDER, PARAMFOLDER, SKIMTIME, SKIMDISTANCE, LINKS, NODES, ZONES, SEGS, \
#                 COMMODITYMATRIX, PARCELNODES, PARCELS_PER_HH, PARCELS_PER_EMPL, PARCELS_MAXLOAD, PARCELS_DROPTIME, \
#                 PARCELS_SUCCESS_B2C, PARCELS_SUCCESS_B2B, PARCELS_GROWTHFREIGHT, \
#                 YEARFACTOR, NUTSLEVEL_INPUT, \
#                 IMPEDANCE_SPEED, \
#                 SHIPMENTS_REF, SELECTED_LINKS,\
#                 LABEL, \
#                 MODULES]
    
#         varStrings = ["INPUTFOLDER", "OUTPUTFOLDER", "PARAMFOLDER", "SKIMTIME", "SKIMDISTANCE", "LINKS", "NODES", "ZONES", "SEGS", \
#                       "COMMODITYMATRIX", "PARCELNODES", "PARCELS_PER_HH", "PARCELS_PER_EMPL", "PARCELS_MAXLOAD", "PARCELS_DROPTIME", \
#                       "PARCELS_SUCCESS_B2C", "PARCELS_SUCCESS_B2B",  "PARCELS_GROWTHFREIGHT", \
#                       "YEARFACTOR", "NUTSLEVEL_INPUT", \
#                       "IMPEDANCE_SPEED", \
#                       "SHIPMENTS_REF", "SELECTED_LINKS", \
#                       "LABEL", \
#                       "MODULES"]
         
#         varDict = {}

#%%
print('Starting Parcel Generation')
actually_run_module(args)


print('Parcel Generation Completed')




#         for i in range(len(args)):
#             varDict[varStrings[i]] = args[i]
            
#         # Run the module
#         main(varDict)

#%%





