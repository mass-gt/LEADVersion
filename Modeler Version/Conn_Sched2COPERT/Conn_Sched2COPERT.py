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
            params_file = open(f'{datapath}/Input/Params_Conn_Sched2COPERT.txt')
            
            # This are the defaults, might need to change for console run!!!
            varDict['LABEL'	]			= 'MVP3b'				
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ 'Input' +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ 'Output' +'/'			
            
            varDict['ParcelActivity']              = varDict['INPUTFOLDER'] + "ParcelSchedule_MVP3b_TotalUrbanDelivery.csv"    

            varDict['Weather'] 		= varDict['INPUTFOLDER']     + "Weather.csv"


            
            
            
            
        else:  # This is the part for line cod execution
            locationparam = f'{datapath}'+'/' + sys.argv[2] +'/' + sys.argv[4]
            params_file = open(locationparam)
            varDict['LABEL'	]			= sys.argv[1]				
            varDict['DATAPATH']			= datapath							
            varDict['INPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[2] +'/' 				
            varDict['OUTPUTFOLDER']		= f'{datapath}'+'/'+ sys.argv[3] +'/'			
            
            varDict['ParcelActivity']              = varDict['INPUTFOLDER'] + sys.argv[5]     

            varDict['Weather'] 		= varDict['INPUTFOLDER'] + sys.argv[6] #'skimTijd_new_REF.mtx' 		
	

           
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


Weather = pd.read_csv(varDict['Weather'])
ParcelTrips = pd.read_csv(varDict['ParcelActivity'] )


filename ="Input_COPERT_" + varDict['LABEL'] +".xlsx"
filenameEV ="Input_EVModel_" + varDict['LABEL'] +"EV.xlsx"


#%% Generate sheet data

'''
NEEDED SHEETS: 
SHEETS: Index of the sheets in the excel file
STOCK: Cars invovled in delivery    # Doubt: if we assign only 1 car all the km??? -> from param file
MEAN_ACTIVITY: Kilometres per car type and unit stock  -> from scheduler
URBAN_OFF_PEAK_SPEED: Average speed off peak per car type -> param file
URBAN_PEAK_SPEED:  Average speed on peak per car type -> param file
URBAN_OFF_PEAK_SHARE: Share of kms done off peak -> from sched
URBAN_PEAK_SHARE: Share of kms done on peak -> from sched
MIN_TEMPERATURE: Min temperature per month -> separate input
MAX_TEMPERATURE: Max temperature per month -> separate input
HUMIDITY: Average humidity per month -> separate input

'''
Year = int(varDict['Year'])



MEAN_ACTIVITY = ParcelTrips.groupby(['VehType'])["TourDist"].sum()
MEAN_ACTIVITY = MEAN_ACTIVITY.reset_index()
MEAN_ACTIVITY.columns = ["Category",Year ]

# Get peak and off peak

Activity_Peak = ParcelTrips[((ParcelTrips["TourDepTime"] >varDict["PeakHourMorningStart"] ) & (ParcelTrips["TourDepTime"] <varDict["PeakHourMorningFinish"] )) |((ParcelTrips["TourDepTime"] >varDict["PeakHourAfternoonStart"] ) & (ParcelTrips["TourDepTime"] <varDict["PeakHourAfternoonFinish"] )) ]
Peak_ACTIVITY = Activity_Peak.groupby(['VehType'])["TourDist"].sum()
Peak_ACTIVITY = Peak_ACTIVITY.reset_index()
Peak_ACTIVITY.columns = ["Category","Peak" ]

# Peak_ACTIVITY = pd.merge(Peak_ACTIVITY,MEAN_ACTIVITY  )
# Peak_ACTIVITY["Peak%"] =round( Peak_ACTIVITY["Peak"] / Peak_ACTIVITY[Year] *100,2)
# Peak_ACTIVITY["OffPeak%"] =100-Peak_ACTIVITY["Peak%"]



# Get vehicle types

Vehiclekms ={}
VehiclePeak ={}



for i,v in enumerate(varDict["VehicleOffPeakSpeed"]):
    Vehiclekms[v] = 0
    VehiclePeak[v] =0

# Convert vehicle use from MASS GT into COPERT vehicles
for i,v in enumerate(varDict["VehicleType"]):
    for j,w in enumerate (varDict["VehicleType"][v]):
        kmsTot = MEAN_ACTIVITY.loc[MEAN_ACTIVITY['Category'] == v]
        KmsPeak = Peak_ACTIVITY.loc[Peak_ACTIVITY['Category'] == v]
        if kmsTot.empty:
            kmsTot = 0
        else:
            kmsTot = kmsTot.iloc[0][Year]
        if KmsPeak.empty:
            KmsPeak = 0
        else:
            KmsPeak = KmsPeak.iloc[0]["Peak"]            
            
            
        Vehiclekms[w] += kmsTot * varDict["VehicleType"][v][w]
        Vehiclekms[w] = round(Vehiclekms[w],2)
        VehiclePeak[w] += KmsPeak * varDict["VehicleType"][v][w]
        VehiclePeak[w] = round(VehiclePeak[w],2)

VehiclePeakPerc ={}
VehicleoffPeakPerc ={}

# Generate peak %
for i,v in enumerate(Vehiclekms):
    Per = round(VehiclePeak[v] / (Vehiclekms[v]+0.0000001)*100,2)
    VehiclePeakPerc[v] = Per 
    VehicleoffPeakPerc[v] =100-   Per   
        
        
        
        
Category =[]
Fuel =[]
Segment = []
EuroStandard = []
Activity = []
UrbanOffPeakSpeed = []
UrbanPeakSpeed = []
UrbanOffPeakShare = []
UrbanPeakShare = []


# MeanActDict = {}

for i,v in enumerate(Vehiclekms):
    # print(v)
    Category.append (varDict["VehicleCat"][v])
    Fuel.append   (varDict["VehicleFuel"][v])
    Segment.append (varDict["VehicleSegment"][v])
    if  v == "Bike":
        EuroStandard.append("0")
    elif v == "Hybrid":
        EuroStandard.append (str(varDict["VehicleEuroStand"][v][0]))
    elif v == "Electric":
        EuroStandard.append("0")
    else:
         EuroStandard.append (varDict["VehicleEuroStand"][v])
    Activity.append(Vehiclekms[v])
    UrbanOffPeakSpeed.append (varDict["VehicleOffPeakSpeed"][v])
    UrbanPeakSpeed.append (varDict["VehiclePeakSpeed"][v])
    UrbanOffPeakShare.append (VehicleoffPeakPerc[v])
    UrbanPeakShare.append (VehiclePeakPerc[v])
    
    
    
    # MeanActDict[v] = [Category,Fuel,Segment,EuroStandard,VehicleTypes[v]]
    
    

ACTIVITY = pd.DataFrame(columns=['Category','Fuel','Segment','Euro Standard'])
ACTIVITY['Category'] = Category
ACTIVITY['Fuel'] = Fuel
ACTIVITY['Segment'] = Segment
ACTIVITY['EuroStandard'] = EuroStandard
ACTIVITY[Year] =  Activity

MEAN_ACTIVITY = ACTIVITY[ACTIVITY["Category"] != 0]

STOCK = ACTIVITY
STOCK[Year] = 1
STOCK = STOCK[STOCK["Category"] != 0]

URBAN_OFF_PEAK_SPEED=ACTIVITY
URBAN_OFF_PEAK_SPEED[Year] = UrbanOffPeakSpeed
URBAN_OFF_PEAK_SPEED = URBAN_OFF_PEAK_SPEED[URBAN_OFF_PEAK_SPEED["Category"] != 0]

URBAN_PEAK_SPEED=ACTIVITY
URBAN_PEAK_SPEED[Year] = UrbanPeakSpeed
URBAN_PEAK_SPEED = URBAN_PEAK_SPEED[URBAN_PEAK_SPEED["Category"] != 0]

URBAN_OFF_PEAK_SHARE = ACTIVITY
URBAN_OFF_PEAK_SHARE[Year] = UrbanOffPeakShare
URBAN_OFF_PEAK_SHARE = URBAN_OFF_PEAK_SHARE[URBAN_OFF_PEAK_SHARE["Category"] != 0]

URBAN_PEAK_SHARE = ACTIVITY
URBAN_PEAK_SHARE[Year] = UrbanPeakShare
URBAN_PEAK_SHARE = URBAN_PEAK_SHARE[URBAN_PEAK_SHARE["Category"] != 0]


## Weather

MIN_TEMPERATURE = Weather [["Month","Min_Temp"]]
MAX_TEMPERATURE = Weather [["Month","Max_Temp"]]
HUMIDITY = Weather [["Month","Humidity"]]







SheetNames = ["STOCK","MEAN_ACTIVITY","URBAN_OFF_PEAK_SPEED","URBAN_PEAK_SPEED","URBAN_OFF_PEAK_SHARE","URBAN_PEAK_SHARE","MIN_TEMPERATURE","MAX_TEMPERATURE","HUMIDITY"]
SheetVals = ["[n]","[km]","[km/h]","[%]","[%]","[℃]","[℃]","[%]"]
SHEETS =  pd.DataFrame(list(zip(SheetNames, SheetVals)),    columns =['SHEET_NAME', 'Unit'])

# Add hyperlinks to sheets

SheetHyper=[]
for i in SheetNames:
    print(i)
    string = filename +"#"+i+"!A1"
    SheetHyper.append ( "=HYPERLINK("+f'"{string}"'+","+f'"{i}"'+")")
SHEETS =  pd.DataFrame(list(zip(SheetHyper, SheetVals)),    columns =['SHEET_NAME', 'Unit'])


#%% Export for EV model

if varDict ["ElectricEmissions"]:
    MEAN_ACTIVITY = ParcelTrips.groupby(['VehType'])["TourDist"].sum()
    MEAN_ACTIVITY = MEAN_ACTIVITY.reset_index()
    MEAN_ACTIVITY.columns = ["Category",Year ]


EnergyConsumpt =0 

for i,v in enumerate(Vehiclekms):
    kms = Vehiclekms[v]

    if v == "Hybrid":
        EnergyConsumpt+= varDict["VehicleEuroStand"][v][1] *varDict["VehicleEuroStand"][v][2]*kms 

    elif v == "Electric":
        EnergyConsumpt+= varDict["VehicleEuroStand"][v] *kms 
    else:
         EnergyConsumpt+=0

    
out =varDict['OUTPUTFOLDER'] + filenameEV
writer = pd.ExcelWriter(out, engine='xlsxwriter')



EVEMISSIONS = pd.DataFrame([[1,round(EnergyConsumpt,2)]],columns=['Stock','Comsumption Vehicle'])




EVEMISSIONS.to_excel(writer, sheet_name='Template', index=False)
writer.save()
#%% Export to input excel files of copert


out =varDict['OUTPUTFOLDER']+ filename

writer = pd.ExcelWriter(out, engine='xlsxwriter')

SHEETS.to_excel(writer, sheet_name='SHEETS', index=False)
STOCK.to_excel(writer, sheet_name='STOCK', index=False)
MEAN_ACTIVITY.to_excel(writer, sheet_name='MEAN_ACTIVITY', index=False)
URBAN_OFF_PEAK_SPEED.to_excel(writer, sheet_name='URBAN_OFF_PEAK_SPEED', index=False)
URBAN_PEAK_SPEED.to_excel(writer, sheet_name='URBAN_PEAK_SPEED', index=False)
URBAN_OFF_PEAK_SHARE.to_excel(writer, sheet_name='URBAN_OFF_PEAK_SHARE', index=False)
URBAN_PEAK_SHARE.to_excel(writer, sheet_name='URBAN_PEAK_SHARE', index=False)
MIN_TEMPERATURE.to_excel(writer, sheet_name='MIN_TEMPERATURE', index=False)
MAX_TEMPERATURE.to_excel(writer, sheet_name='MAX_TEMPERATURE', index=False)
HUMIDITY.to_excel(writer, sheet_name='HUMIDITY', index=False)


writer.save()




















