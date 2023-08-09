# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:02:37 2021
@author: beren
"""

from __functions__ import read_mtx, read_shape, get_traveltime, get_distance
import pandas as pd
import numpy as np
from scipy import spatial
import math
import os


#OLD!!!
# def getMax (matrix,cols,rows,remove =1):  # If I do the first line and column as the trip ID or parcel ID, then I can remove the lines! If not, add a key that you remove as well when you remvoe the line!
#     # Remove = 1 is that removes the line --> each person takes only 1 parcel per trip!!    
#     maximum = np.amax(matrix)
#     position = np.where(matrix == maximum)

#     pair = {rows.iloc[position[0][0]][0]:cols.iloc[position[1][0]][0]}  # This is the result!
    

#     matrix = np.delete(matrix,position[1][0],1) # Delete parcel (column) from list
#     cols = cols.drop(cols.index[position[1][0]]).reset_index(drop=True)

#     if remove ==1 : # Delete row
#         matrix = np.delete(matrix,position[0][0],0) # Delete parcel (column) from list
#         rows= rows.drop(rows.index[position[0][0]]).reset_index(drop=True)
#     else: # Put zeros
#         matrix[position[0],:] = np.zeros(( len(matrix[position[0],:][0] )))    
    
#     return pair, maximum, matrix, cols,rows

def Willinness2Bring (UtilFunction, unique_id):  # TODO: How to add the variables from the columns
    
    # unique_id is the 
    # row = trips.loc[trips['unique_id']==unique_id]
    # mode = row ['mode']
    # # preced_purpose  = row ['mode']
    # # follow_purpose = row ['mode']
    # age = row ['age']
    # gender = row ['gender']
    # work = row ['work']
    # hh_income = row ['hh_income']
    
    Utility = eval(UtilFunction)
    Prob = 1/(1+np.exp(-Utility))
    return Prob



# position,value,Surplus,Detours,detour,lableParcels,lableTrav = getMax (Surplus,lableParcels,lableTrav,Detours,remove = 1)

def getMax (matrix,cols,rows,othermatrix =np.nan, remove =1):  # If I do the first line and column as the trip ID or parcel ID, then I can remove the lines! If not, add a key that you remove as well when you remvoe the line!
    # Remove = 1 is that removes the line --> each person takes only 1 parcel per trip!!    
    maximum = np.amax(matrix)
    position = np.where(matrix == maximum)
    # valueinothermatrix = othermatrix[position][0]

    pair = {rows.iloc[position[0][0]][0]:cols.iloc[position[1][0]][0]}  # This is the result!
    valueinothermatrix = {rows.iloc[position[0][0]][0]:othermatrix[position][0]}

    matrix = np.delete(matrix,position[1][0],1) # Delete parcel (column) from list
    othermatrix = np.delete(othermatrix,position[1][0],1) # Delete parcel (column) from list
    cols = cols.drop(cols.index[position[1][0]]).reset_index(drop=True)

    if remove ==1 : # Delete row
        matrix = np.delete(matrix,position[0][0],0) # Delete parcel (column) from list
        othermatrix= np.delete(othermatrix,position[0][0],0) # Delete parcel (column) from list
        rows= rows.drop(rows.index[position[0][0]]).reset_index(drop=True)
    else: # Put zeros
        matrix[position[0],:] = np.zeros(( len(matrix[position[0],:][0] )))    
    
    return pair, maximum, matrix,othermatrix,valueinothermatrix, cols,rows


def generate_Utility (UtilityFunct, variables,deterministic=1):  # TODO: How to add the variables from the columns
 
 
     for key,val in variables.items():
             exec(key + '=val')
         
     Utility = eval(UtilityFunct)
     
     if deterministic == 1:
         Utility += np.log(-np.log(np.random.uniform()))#np.random.gumbel()
     
     
     return Utility 
   

def actually_run_module(args):
    # -------------------- Define datapaths -----------------------------------
            
    root    = args[0]
    varDict = args[1]
            
    if root != '':
        root.progressBar['value'] = 0
            
    # Define folders relative to current datapath
    datapath        = varDict['DATAPATH']
    VoT             = varDict['VOT']
    droptime_car    = varDict['PARCELS_DROPTIME_CAR']
    droptime_bike   = varDict['PARCELS_DROPTIME_BIKE']
    droptime_pt     = varDict['PARCELS_DROPTIME_PT']
    # CS_willingness  = varDict['CS_WILLINGNESS']
    CS_willingness  = varDict['CS_BaseBringerWillingess']
    CS_BringerFilter  = varDict['CS_BringerFilter']
    CS_BringerUtility  = varDict['CS_BringerUtility']
    
    
    Pax_Trips      = varDict['Pax_Trips']
    
    
    
    
    
    skims = {'time': {}, 'dist': {}, }
    skims['time']['path'] = varDict['SKIMTIME']
    skims['dist']['path'] = varDict['SKIMDISTANCE']
    for skim in skims:
        skims[skim] = read_mtx(skims[skim]['path'])
        nSkimZones = int(len(skims[skim])**0.5)
        skims[skim] = skims[skim].reshape((nSkimZones, nSkimZones))
        if skim == 'time': skims[skim][6483] = skims[skim][:,6483] = 5000 # data deficiency
        for i in range(nSkimZones): #add traveltimes to internal trips
            skims[skim][i,i] = 0.7 * np.min(skims[skim][i,skims[skim][i,:]>0])
        skims[skim] = skims[skim].flatten()
    skimTravTime = skims['time']; skimDist = skims['dist']
    del skims, skim, i
    timeFac = 1   # The skim time is in hours now!!!
    
    skimTime = {}
    skimTime['car'] = skimTravTime
    skimTime['car_passenger'] = skimTravTime
    skimTime['walk'] = (skimDist / 1000 / 5 * 3600).astype(int)
    skimTime['bike'] = (skimDist / 1000 / 12 * 3600).astype(int)
    skimTime['pt'] = skimTravTime * 2 #https://doi.org/10.1038/s41598-020-61077-0, http://dx.doi.org/10.1016/j.jtrangeo.2013.06.011
    
    zones = read_shape(varDict['ZONES'])
    zones.index = zones['AREANR']
    nZones = len(zones)
    
    zoneDict  = dict(np.transpose(np.vstack( (np.arange(1,nZones+1), zones['AREANR']) )))
    zoneDict  = {int(a):int(b) for a,b in zoneDict.items()}
    invZoneDict = dict((v, k) for k, v in zoneDict.items()) 
        
    #%% Generate bringers supply
    def generate_CS_supply(trips, CS_willingness,CS_BringerFilter): # CS_willingness is the willingness to be a bringer
        
        # Filter out depending on SE. This can be directly done as well 
        for SE_Filter in CS_BringerFilter:
            # print(SE_Filter)
            trips = trips.loc[trips[SE_Filter].isin(CS_BringerFilter[SE_Filter])]
            # print(len(trips))
        
        
        trips['CS_willing'] =np.random.uniform(0,1,len(trips)) < trips['unique_id'].apply(lambda x: Willinness2Bring(CS_willingness,x))       # Willingness a priori. This is the willingness to be surbscribed in the platform
        
    
        # trips['CS_willing'] = np.random.uniform(0,1,len(trips)) < CS_willingness
        trips['CS_eligible'] = (trips['CS_willing'])
        
        tripsCS = trips[(trips['CS_eligible'] == True)]
        tripsCS = tripsCS.drop(['CS_willing', 'CS_eligible' ],axis=1)
        
        # #transform the lyon data into The Hague data  # This is not needed anymore because of albatross data
        # for i, column in enumerate(['origin_x', 'destination_x', 'origin_y', 'destination_y']):
        #     tripsCS[column] = (tripsCS[column]-min(tripsCS[column])) / (max(tripsCS[column]) - min(tripsCS[column]))
        #     if i < 2: tripsCS[column] = tripsCS[column] * (max(zones['X'])-min(zones['X'])) + min(zones['X'])
        #     if i > 1: tripsCS[column] = tripsCS[column] * (max(zones['Y'])-min(zones['Y'])) + min(zones['Y'])
        

        
        coordinates = [((zones.loc[zone, 'X'], zones.loc[zone, 'Y'])) for zone in zones.index]
        tree = spatial.KDTree(coordinates)
        
        tripsCS['O_zone'], tripsCS['D_zone'], tripsCS['travtime'], tripsCS['travdist'], tripsCS['municipality_orig'], tripsCS['municipality_dest'] , tripsCS['BaseUtility']= np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        trips_array = np.array(tripsCS)
        # for i,value in enumerate(tripsCS.columns):  # Sirve para saber los numeros para abaj0
        #     print(i,value)
        
        for traveller in trips_array:
            mode = traveller[21]
            traveller[22] = int(zoneDict[tree.query([(traveller[17], traveller[18])])[1][0]+1]) #orig
            traveller[23] = int(zoneDict[tree.query([(traveller[19], traveller[20])])[1][0]+1]) #dest
            traveller[25] = get_distance(invZoneDict[traveller[22]], invZoneDict[traveller[23]], skimDist, nSkimZones) # in km!
            if mode == 'Car':
                traveller[24] = get_traveltime(invZoneDict[traveller[22]], invZoneDict[traveller[23]], skimTime['car'], nSkimZones, timeFac) # in hours
                traveller[24] = traveller[24] * 60 # in minutes now!

                cost          = traveller[25] * eval(varDict ['Car_CostKM'])
                traveller[28] = generate_Utility (CS_BringerUtility,{'Cost': cost,'Time':traveller[24]})

            elif mode == 'Car as Passenger':
                traveller[24] = get_traveltime(invZoneDict[traveller[22]], invZoneDict[traveller[23]], skimTime['car_passenger'], nSkimZones, timeFac) # in hours
                traveller[24] = traveller[24] * 60 # in minutes now!
                cost   = traveller[25] * eval(varDict ['Car_CostKM'])
                traveller[28] = generate_Utility (CS_BringerUtility,{'Cost': cost,'Time':traveller[24]})

            elif mode == 'Walking or Biking':
                traveller[24] = 60 * traveller[25] / varDict['WalkBikeSpeed']  # CHANGE FOR CORRECT BIKE SPEED (AND CORRECT UNITS)
                cost = 0
                traveller[28] = generate_Utility (CS_BringerUtility,{'Cost': cost,'Time':traveller[24]})
            elif mode == 'walk':
                traveller[24] =  60 *traveller[25] / 1  # CHANGE FOR CORRECT walk SPEED (AND CORRECT UNITS)
                cost  = 0
                traveller[28] = generate_Utility (CS_BringerUtility,{'Cost': cost,'Time':traveller[24]})

            else: 
                traveller[19] = np.nan
            traveller[26] = zones.loc[traveller[22], 'GEMEENTEN']
            traveller[27] = zones.loc[traveller[23], 'GEMEENTEN']
        
        
        tripsCS = pd.DataFrame(trips_array, columns=tripsCS.columns)
        
        return tripsCS
    

    trips = pd.read_csv(Pax_Trips, sep = ',', )
    global tripsCS
    print("generating bringer supply")
    tripsCS = generate_CS_supply(trips, CS_willingness,CS_BringerFilter)
    tripsCS['shipping'] = np.nan
    # print('test')
    
    DirCS_Parcels = f"{varDict['OUTPUTFOLDER']}Parcels_CS_{varDict['LABEL']}.csv"
    parcels = pd.read_csv(DirCS_Parcels)
    print('Number of parcels loaded in the Crowdshipping module: ', len(parcels))
    parcels["traveller"],parcels["trip"], parcels["detour"], parcels["compensation"] = '', np.nan, np.nan,0
    
   #%% Checking whether the parcels will use CS with a choice model
    def get_compensation(dist_parcel_trip): # This could potentially have more vars!
        #compensation = math.log( (dist_parcel_trip) + 2)
        compensation = eval(varDict['CS_COMPENSATION'])
        return compensation
        
    
    parcels['parceldistance'] = parcels.apply(lambda x: get_distance(x.O_zone,x.D_zone, skimDist, nSkimZones), axis=1) 
    parcels['compensation'] = parcels.apply(lambda x: get_compensation(x.parceldistance), axis=1)
    parcels['cost'] = parcels['compensation'] * (1+ varDict['PlatformComission']+ varDict['CS_Costs'])
    parcels['CS_comission'] = parcels['cost'] *  varDict['PlatformComission']
    parcels['CS_deliveryChoice'] = parcels.apply(lambda x: generate_Utility(varDict['CS_Willingess2Send'],{'Cost': x.cost,'TradCost' : varDict['TradCost']}), axis=1)
    parcels['CS_deliveryChoice'] = parcels['CS_deliveryChoice'] > 0   # I already simulated the gumbell distribution, so the utility is the actual choice!!!!!!!

    # parcels2traditional = parcels [parcels['CS_deliveryChoice'] == False]  # Take out the parcels to be sent by traditional services!!!
    # parcels             = parcels [parcels['CS_deliveryChoice'] == True]
    
    
    #%% Matching of parcels and travellers


    print("Matching parcels with bringers")
    if varDict['CS_ALLOCATION'] == 'MinimumDistance':  # This is the old approach
        
        for index, parcel in parcels.iterrows():
            parc_orig = parcel['O_zone']
            parc_dest = parcel['D_zone']
            parc_orig_muni = zones.loc[parc_orig, 'GEMEENTEN']
            parc_dest_muni = zones.loc[parc_dest, 'GEMEENTEN']
            parc_dist = get_distance(parc_orig, parc_dest, skimDist, nSkimZones)   # skimDist[(parc_orig-1),(parc_dest-1)] / 1000
            # compensation = get_compensation(parc_dist)
            # parcel['compensation'] = compensation
            compensation = parcel['compensation']
            Minimizing_dict = {}
            filtered_trips = tripsCS[((parc_dist / tripsCS['travdist'] < 1) & 
                                      (tripsCS['shipping'].isnull()) & 
                                      ((parc_orig_muni == tripsCS['municipality_orig']) | (parc_orig_muni == tripsCS['municipality_dest']) |
                                       (parc_dest_muni == tripsCS['municipality_orig']) | (parc_dest_muni == tripsCS['municipality_dest'])))]
            for i, traveller in filtered_trips.iterrows():
                VoT = eval (varDict['VOT'])  # In case I will do the VoT function of the traveller sociodems/purpose, etc
                trav_orig = traveller['O_zone']
                trav_dest = traveller['D_zone']
                mode = traveller['Mode']
                trip_time = traveller['travtime']
                trip_dist = traveller['travdist']
                if mode in ['car','Car']: 
                    CS_pickup_time = droptime_car
                    mode = 'car'
                    CS_pickup_time = droptime_car
                    time_traveller_parcel   = get_traveltime(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime['car'], nSkimZones, timeFac) # These are car/can TTs!!
                    time_parcel_trip        = get_traveltime(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime['car'], nSkimZones, timeFac) # These are car/can TTs!!
                    time_customer_end       = get_traveltime(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime['car'], nSkimZones, timeFac) # These are car/can TTs!!
                    time_traveller_parcelT   = 60 * time_traveller_parcel # Result in minutes (if timefact = 1)
                    time_parcel_tripT        = 60 * time_parcel_trip   # Result in minutes (if timefact = 1)
                    time_customer_endT      = 60 * time_customer_end # Result in minutes (if timefact = 1)
                    
                    
                    # dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime[mode], nSkimZones) # These are car/can TTs!!
                    # dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                    # dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                    dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimDist, nSkimZones) # These are car/can TTs!!
                    dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimDist, nSkimZones) # These are car/can TTs!!
                    dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimDist, nSkimZones) # These are car/can TTs!!
                    time_traveller_parcel   = 60 * dist_traveller_parcel  /  varDict['CarSpeed'] 
                    time_parcel_trip        = 60 * dist_parcel_trip       /  varDict['CarSpeed'] # Result in minutes (if timefact = 1)
                    time_customer_end       = 60 * dist_customer_end     /   varDict['CarSpeed'] # Result in minutes (if timefact = 1)
                    
                    time_traveller_parcel = max(time_traveller_parcelT,time_traveller_parcel)
                    time_parcel_trip= max(time_parcel_tripT,time_parcel_trip)
                    time_customer_end= max(time_customer_endT,time_customer_end)
                if mode in ['bike', 'car_passenger','Walking or Biking']: 
                    CS_pickup_time = droptime_bike
                    dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimDist, nSkimZones) # These are car/can TTs!!
                    dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimDist, nSkimZones) # These are car/can TTs!!
                    dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimDist, nSkimZones) # These are car/can TTs!!
                    time_traveller_parcel   = 60 * dist_traveller_parcel  /  varDict['WalkBikeSpeed']  # Result in minutes (if timefact = 1)
                    time_parcel_trip        = 60 * dist_parcel_trip       /  varDict['WalkBikeSpeed'] # Result in minutes (if timefact = 1)
                    time_customer_end       = 60 * dist_customer_end     /   varDict['WalkBikeSpeed'] # Result in minutes (if timefact = 1)
                if mode in ['walk', 'pt']: 
                    CS_pickup_time = droptime_pt
                
                # time_traveller_parcel   = get_traveltime(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime[mode], nSkimZones, timeFac)
                # time_parcel_trip        = get_traveltime(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime[mode], nSkimZones, timeFac)
                # time_customer_end       = get_traveltime(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime[mode], nSkimZones, timeFac)
                CS_trip_time = (time_traveller_parcel + time_parcel_trip + time_customer_end)
                CS_detour_time = CS_trip_time - trip_time
                
                if ((CS_detour_time + CS_pickup_time * 2)/3600) == 0: CS_detour_time += 1 #prevents /0 eror
                compensation_time =  compensation / ((CS_detour_time + CS_pickup_time * 2)/3600)
                if compensation_time > VoT:
                    dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimDist, nSkimZones)
                    dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimDist, nSkimZones)
                    dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimDist, nSkimZones)
                    CS_trip_dist = (dist_traveller_parcel + dist_parcel_trip + dist_customer_end)
                    CS_surplus   = compensation + VoT * CS_detour_time/3600 # Is VOT in hours? Is CS_detour time in seconds?
                    if varDict ['CS_BringerScore'] == 'Surplus':    # Is it bad practive to bring the varDict into the code?
                        CS_Min = (-1)* CS_surplus  # The -1 is to minimize the surplus
                    elif varDict ['CS_BringerScore'] == 'Min_Detour':
                        CS_Min = round(CS_trip_dist - trip_dist, 5)
                    elif varDict ['CS_BringerScore'] == 'Min_Time':
                        CS_Min = round(((CS_detour_time + CS_pickup_time * 2)/3600) - trip_time, 5)
                    Minimizing_dict[f"{traveller['person_id']}_{traveller['person_trip_id']}"] = CS_Min
                
            if Minimizing_dict:  # The traveler that has the lowest detour gets the parcel
                traveller = min(Minimizing_dict, key=Minimizing_dict.get)
                parcels.loc[index, 'traveller'] = traveller
                parcels.loc[index, 'detour'] = Minimizing_dict[traveller]
                parcels.loc[index, 'compensation'] = compensation
                parcels.loc[index, 'Mode'] = filtered_trips.loc[filtered_trips["unique_id"]==traveller,'Mode'].iloc[0]
                parcels.loc[index, 'unique_id'] = traveller
                # person, trip = traveller.split('_')
                # person = int(person); trip = int(trip)
                # # print(traveller)
                # tripsCS.loc[((tripsCS['person_id'] == person) & (tripsCS['person_trip_id'] == trip)), 'shipping'] = parcels.loc[index, 'Parcel_ID'] # Are we saving the trips CS?
        
    elif varDict['CS_ALLOCATION'] == 'best2best':
        Surpluses = np.zeros((len(tripsCS),len(parcels))) 
        Detours   = np.zeros((len(tripsCS),len(parcels))) 
        # Detours   = np.zeros((len(tripsCS),len(parcels))) 
        
        # comp = []
        print("Starting Utility evaluation")
        for index, parcel in parcels.iterrows():
            parc_orig = parcel['O_zone']
            parc_dest = parcel['D_zone']
            parc_orig_muni = zones.loc[parc_orig, 'GEMEENTEN']
            parc_dest_muni = zones.loc[parc_dest, 'GEMEENTEN']
            parc_dist = get_distance(parc_orig, parc_dest, skimDist, nSkimZones)   # skimDist[(parc_orig-1),(parc_dest-1)] / 1000
            # compensation = get_compensation(parc_dist)
            # comp.append(compensation)
            # parcels.iloc[index]['compensation'] = compensation
            
            compensation = parcel['compensation']
            # This is a preventive filter to have less suitable trips. Nimber does this by doing a "fat line" of routes. 
            #TODO 
            filtered_trips = tripsCS[((parc_dist / tripsCS['travdist'] < 1) & # Parcel dist has to be lower than the actual trip, why?? TODO 
                                      (tripsCS['shipping'].isnull()) & 
                                      ((parc_orig_muni == tripsCS['municipality_orig']) | (parc_orig_muni == tripsCS['municipality_dest']) |
                                        (parc_dest_muni == tripsCS['municipality_orig']) | (parc_dest_muni == tripsCS['municipality_dest'])))]
            
            if parcel['CS_deliveryChoice']:
                
                for i, traveller in filtered_trips.iterrows():
                    # print(traveller['PossibleParcels'] )
                    # VoT = eval (varDict['VOT'])  # In case I will do the VoT function of the traveller sociodems/purpose, etc
                    trav_orig = traveller['O_zone']
                    trav_dest = traveller['D_zone']
                    mode = traveller['Mode']
                    trip_time = traveller['travtime']
                    trip_dist = traveller['travdist']
                    if mode in ['Car']: 
                        CS_pickup_time = droptime_car
                        time_traveller_parcel   = get_traveltime(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime['car'], nSkimZones, timeFac) # These are car/can TTs!!
                        time_parcel_trip        = get_traveltime(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime['car'], nSkimZones, timeFac) # These are car/can TTs!!
                        time_customer_end       = get_traveltime(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime['car'], nSkimZones, timeFac) # These are car/can TTs!!
                        time_traveller_parcelT   = 60 * time_traveller_parcel # Result in minutes (if timefact = 1)
                        time_parcel_tripT        = 60 * time_parcel_trip   # Result in minutes (if timefact = 1)
                        time_customer_endT      = 60 * time_customer_end # Result in minutes (if timefact = 1)
                        
                        
                        # dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        # dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        # dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimDist, nSkimZones) # These are car/can TTs!!
                        dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimDist, nSkimZones) # These are car/can TTs!!
                        dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimDist, nSkimZones) # These are car/can TTs!!
                        time_traveller_parcel   = 60 * dist_traveller_parcel  /  varDict['CarSpeed'] 
                        time_parcel_trip        = 60 * dist_parcel_trip       /  varDict['CarSpeed'] # Result in minutes (if timefact = 1)
                        time_customer_end       = 60 * dist_customer_end     /   varDict['CarSpeed'] # Result in minutes (if timefact = 1)
                        
                        time_traveller_parcel = max(time_traveller_parcelT,time_traveller_parcel)
                        time_parcel_trip= max(time_parcel_tripT,time_parcel_trip)
                        time_customer_end= max(time_customer_endT,time_customer_end)
                        
                        CS_TravelCost           = eval(varDict ['Car_CostKM'])
                    if mode in ['Walking or Biking', 'Car as Passenger']: CS_pickup_time = droptime_bike
                    if mode in ['Walking or Biking']:
                        # dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        # dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        # dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimDist, nSkimZones) # These are car/can TTs!!
                        dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimDist, nSkimZones) # These are car/can TTs!!
                        dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimDist, nSkimZones) # These are car/can TTs!!
                        time_traveller_parcel   = 60 * dist_traveller_parcel  /  varDict['WalkBikeSpeed']  # Result in minutes (if timefact = 1)
                        time_parcel_trip        = 60 * dist_parcel_trip       /  varDict['WalkBikeSpeed'] # Result in minutes (if timefact = 1)
                        time_customer_end       = 60 * dist_customer_end     /   varDict['WalkBikeSpeed'] # Result in minutes (if timefact = 1)
                        
                        # Change this to make sure I don't have negative time detours! (sometimes it happens that they have less travel distances)
                        
                        time_traveller_parcelT  = 60* get_traveltime(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime['car'], nSkimZones, timeFac) * varDict['CarSpeed']/varDict['WalkBikeSpeed']
                        time_parcel_tripT       = 60*get_traveltime(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime['car'], nSkimZones, timeFac) * varDict['CarSpeed']/varDict['WalkBikeSpeed']
                        time_customer_endT      = 60*get_traveltime(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime['car'], nSkimZones, timeFac)* varDict['CarSpeed']/varDict['WalkBikeSpeed']
                        
                        time_traveller_parcel = max(time_traveller_parcelT,time_traveller_parcel)
                        time_parcel_trip= max(time_parcel_tripT,time_parcel_trip)
                        time_customer_end= max(time_customer_endT,time_customer_end)
                        
                        CS_TravelCost           = 0
                    if mode in ['walk', 'pt']: CS_pickup_time = droptime_pt
                    if mode in ['walk']:
                        # dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        # dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        # dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimTime[mode], nSkimZones) # These are car/can TTs!!
                        dist_traveller_parcel   = get_distance(invZoneDict[trav_orig], invZoneDict[parc_orig], skimDist, nSkimZones) # These are car/can TTs!!
                        dist_parcel_trip        = get_distance(invZoneDict[parc_orig], invZoneDict[parc_dest], skimDist, nSkimZones) # These are car/can TTs!!
                        dist_customer_end       = get_distance(invZoneDict[parc_dest], invZoneDict[trav_dest], skimDist, nSkimZones) # These are car/can TTs!!
                        time_traveller_parcel   =  60 *dist_traveller_parcel  / 1  # CHANGE FOR CORRECT walk SPEED (AND CORRECT UNITS)
                        time_parcel_trip        =  60 *dist_parcel_trip       / 1  # CHANGE FOR CORRECT walk SPEED (AND CORRECT UNITS)
                        time_customer_end       =  60 *dist_customer_end     / 1  # CHANGE FOR CORRECT walk SPEED (AND CORRECT UNITS)                
                        CS_TravelCost           = 0                
                    
                    
                    
                    CS_trip_dist = (dist_traveller_parcel + dist_parcel_trip + dist_customer_end)
                    CS_trip_time = (time_traveller_parcel + time_parcel_trip + time_customer_end)
                    CS_detour_time = CS_trip_time + 2 * CS_pickup_time - trip_time
                    CS_detour_dist = CS_trip_dist - trip_dist
                    
                    NetCompensation = compensation - CS_detour_dist * CS_TravelCost # Is this better approx? Do people compare just the compensation with the time or the pocket compensation??
                    
                    
                    
                    if ((CS_detour_time + CS_pickup_time * 2)) == 0: CS_detour_time += 1 #prevents /0 eror
                    compensation_time =  NetCompensation / ((CS_detour_time + CS_pickup_time * 2)/3600)
                    
                    Util_PickUp = generate_Utility (CS_BringerUtility,{'Cost': CS_TravelCost*CS_trip_dist-compensation,'Time':CS_trip_time}) #+80# This is a provisional number so there aren't that many parcels that are eligible until we find the correct equation
                    
                    TravUtil = generate_Utility (CS_BringerUtility,{'Cost': (CS_TravelCost*traveller['travdist']),'Time':traveller['travtime']})
                    # Surplus1   = Util_PickUp-traveller['BaseUtility']
                    Surplus = Util_PickUp - TravUtil
                    # print(generate_Utility (CS_BringerUtility,{'Cost': -NetCompensation,'Time':CS_trip_time})- generate_Utility (CS_BringerUtility,{'Cost': (CS_TravelCost*traveller['travdist']),'Time':traveller['travtime']}))
                    # print(Surplus1-Surplus)
                    Detour    = CS_trip_dist - traveller['travdist']
                    # print(Surplus)
                    
                    #TODO: para estar seguros q no da ese error de la distancia negativa:
                    if traveller['travtime'] >=   CS_trip_time    :  # Disqualify people that save time
                        # print(Surplus)    
                        Surplus = 0
                    if Detour<0:  # Disqualify people that save distance
 
                        Surplus = 0
                    if Surplus>0:
                        # print(Util_PickUp,traveller['BaseUtility'],NetCompensation,CS_trip_time-traveller['travtime'],Surplus,Surplus/0.045)
                        # print(Surplus,Detour)
                        # print(NetCompensation)
                        Surpluses[i,index] = Surplus
                        Detours  [i,index] = Detour
                        # np.amax(Surpluses)
                    
            # print("parcel ", index, " from ", len(parcels))




    
        lableTrav = tripsCS[['unique_id']].reset_index(drop=True)
        lableParcels = parcels[['Parcel_ID']].reset_index(drop=True)
        
        Surplus = Surpluses
        
        value = 1
        matches ={}
        distances = {}
        discount ={}
        if varDict['CS_UtilDiscount'] == 'SecondBest':
            discountParam = 'SecondBestDiscount'
        elif  varDict['CS_UtilDiscount'] == 'none':
            discountParam = 'none'
        matrix = Surplus
        rows = lableTrav
        cols = lableParcels
        print("Starting matching")
        # print('get max')
        while value != 0:
            position,value,Surplus,Detours,detour,lableParcels,lableTrav = getMax (Surplus,lableParcels,lableTrav,Detours,remove = 1)

            if value !=0:
                matches.update(position)
                distances.update(detour)
            # print(position)
            # print(detour)
            # print(value)
        matches_trips = matches
        matches_parcels =  {v: k for k, v in matches.items()}
        
        
        
        
        parcels['trip'] = parcels['Parcel_ID'].map(matches_parcels)
        parcels['detour'] = parcels['trip'].map(distances)
        

        tripsMode = trips[["unique_id","Mode"]]
        #Get the mode and add it to the parcel
        
        print(" ")
        print(" Parcels just out of the matching", len(parcels))
        parcels =  pd.merge(parcels,tripsMode,left_on='trip', right_on='unique_id',how='left')
        print(" ")
        print(" Parcels after the merge", len(parcels))
        
        parcels[parcels ['CS_deliveryChoice']==False] ['trip'] = 'NaN'
        
        UnmatchedParcels = parcels.drop(parcels[parcels['trip'].notna()].index)
        MatchedParcels  =parcels.drop(parcels[parcels['trip'].isna()].index)
        
        
        MatchedParcels['traveller'] = MatchedParcels['trip'].apply(lambda x: x[0:-2])
        UnmatchedParcels['traveller'] = UnmatchedParcels['trip']
        
        
        parcels = MatchedParcels.append(UnmatchedParcels)
        
        
    #### THE CROWDSHIPPING TRIP DATAFRAME HAS NOT BEEN UPDATED WITH THE NEW TRIPS!!!!!!

    # person, trip = traveller.split('_')
    # person = int(person); trip = int(trip)
    # # print(traveller)
    # tripsCS.loc[((tripsCS['person_id'] == person) & (tripsCS['person_trip_id'] == trip)), 'shipping'] = parcels.loc[index, 'Parcel_ID'] # Are we saving the trips CS?

    


    print(" ")  

    print(" Parcels to be exported", len(parcels))

    parcels.to_csv(f"{varDict['OUTPUTFOLDER']}Parcels_CS_matched_{varDict['LABEL']}.csv", index=False)
    tripsCS.to_csv(f"{varDict['OUTPUTFOLDER']}TripsCS_{varDict['LABEL']}.csv", index=False)
    print("Crowdshipping allocation done")
#%% Run module on itself
if __name__ == '__main__':
    cwd = os.getcwd()
    datapath = cwd.replace('Code', '')

    def generate_args():
        varDict = {}
        '''FOR ALL MODULES'''
        varDict['LABEL']                = 'REF'
        varDict['DATAPATH']             = datapath
        varDict['INPUTFOLDER']          = f'{datapath}Input/Mass-GT/'
        varDict['OUTPUTFOLDER']         = f'{datapath}Output/Mass-GT/'
        varDict['PARAMFOLDER']	        = f'{datapath}Parameters/Mass-GT/'
        
        varDict['SKIMTIME']             = varDict['INPUTFOLDER'] + 'skimTijd_new_REF.mtx'
        varDict['SKIMDISTANCE']         = varDict['INPUTFOLDER'] + 'skimAfstand_new_REF.mtx'
        varDict['ZONES']                = varDict['INPUTFOLDER'] + 'Zones_v4.shp'
        varDict['SEGS']                 = varDict['INPUTFOLDER'] + 'SEGS2020.csv'
        varDict['PARCELNODES']          = varDict['INPUTFOLDER'] + 'parcelNodes_v2.shp'
        varDict['CEP_SHARES']           = varDict['INPUTFOLDER'] + 'CEPshares.csv'
        
        '''FOR CROWDSHIPPING MATCHING MODULE'''
        varDict['CS_WILLINGNESS']       = 0.2
        varDict['VOT']                  = 9.00
        varDict['PARCELS_DROPTIME_CAR'] = 120
        varDict['PARCELS_DROPTIME_BIKE']= 60 #and car passenger
        varDict['PARCELS_DROPTIME_PT']  = 0 #and walk
        varDict['TRIPSPATH']            = f'{datapath}Drive Lyon/'
        
        args = ['', varDict]
        return args, varDict
        
    args, varDict = generate_args()
    actually_run_module(args)