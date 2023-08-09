"""Processing module.
"""
from logging import getLogger
from time import time
from json import dump
from os.path import join

import numpy as np
import pandas as pd

from .utils import (read_shape, read_mtx, get_distance,
                    cluster_parcels, create_schedules)


logger = getLogger("parceltourformation.proc")


def execute(cfg: dict, hub2hub=False) -> None:
    """_summary_

    :param cfg: _description_
    :type cfg: dict
    :param hub2hub: _description_, defaults to False
    :type hub2hub: bool, optional
    """
    dropOffTimeSec = cfg['PARCELS_DROPTIME']
    maxVehicleLoad = cfg['PARCELS_MAXLOAD']
    ParcelFile =  cfg['PARCELS']

    maxVehicleLoad = int(maxVehicleLoad)

    if hub2hub:
        dropOffTimeSec = cfg['PARCELS_DROPTIME_Hub2Hub']
        maxVehicleLoad = cfg['PARCELS_MAXLOAD_Hub2Hub']
        ParcelFile = cfg['PARCELS_HUB2HUB']

        maxVehicleLoad = int(maxVehicleLoad)

    doCrowdShipping = (str(cfg['CROWDSHIPPING']).upper() == 'TRUE')

    exportTripMatrix = True

    # --------------------------- Import data----------------------------------
    logger.info('Importing data...')
    parcels = pd.read_csv(ParcelFile)

    parcelNodes, coords = read_shape(cfg['PARCELNODES'], return_geometry=True)
    parcelNodes['X'] = [coords[i]['coordinates'][0] for i in range(len(coords))]
    parcelNodes['Y'] = [coords[i]['coordinates'][1] for i in range(len(coords))]

    # EDIT 6-10: SHADOW PARCEL NODES - start
    parcelNodes_shadow = parcelNodes.copy()
    parcelNodes_shadow['id'] = parcelNodes_shadow['id']+1000
    parcelNodes = parcelNodes.append(parcelNodes_shadow, ignore_index=True,sort=False)

    # EDIT 6-10: SHADOW PARCEL NODES - end'''
    parcelNodes['id'] = parcelNodes['id'].astype(int)
    parcelNodes.index = parcelNodes['id']
    parcelNodes = parcelNodes.sort_index()
    parcelNodesCEP = {}
    for i in parcelNodes.index:
        parcelNodesCEP[parcelNodes.at[i,'id']] = parcelNodes.at[i,'CEP']

    zones = read_shape(cfg['ZONES'])
    zones = zones.sort_values('AREANR')
    zones.index = zones['AREANR']

    supCoordinates = pd.read_csv(cfg['SUP_COORDINATES'], sep=',')
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
    zoneDict = dict(np.transpose(np.vstack( (np.arange(1, nIntZones+1), zones['AREANR']) )))
    zoneDict = {int(a): int(b) for a, b in zoneDict.items()}
    for i in range(nSupZones):
        zoneDict[nIntZones+i+1] = 99999900 + i + 1
    invZoneDict = dict((v, k) for k, v in zoneDict.items())

    # Change zoning to skim zones which run continuously from 0
    parcels['X'] = [zonesX[x] for x in parcels['D_zone'].values]
    parcels['Y'] = [zonesY[x] for x in parcels['D_zone'].values]
    parcels['D_zone'] = [invZoneDict[x] for x in parcels['D_zone']]
    parcels['O_zone'] = [invZoneDict[x] for x in parcels['O_zone']]
    parcelNodes['skim_zone'] = [invZoneDict[x] for x in parcelNodes['AREANR']]

    # System input for scheduling
    parcelDepTime = np.array(pd.read_csv(cfg['DEPARTURE_TIME_PARCELS_CDF']).iloc[:,1])
    dropOffTime = dropOffTimeSec/3600
    skimTravTime = read_mtx(cfg['SKIMTIME'])
    skimDistance = read_mtx(cfg['SKIMDISTANCE'])
    nZones = int(len(skimTravTime)**0.5)

    # Intrazonal impedances
    skimTravTime = skimTravTime.reshape(nZones,nZones)
    skimTravTime[:, 6483] = 2000.0
    for i in range(nZones):
        skimTravTime[i, i] = 0.7 * np.min(skimTravTime[i, skimTravTime[i, :]>0])
    skimTravTime = skimTravTime.flatten()
    skimDistance = skimDistance.reshape(nZones, nZones)
    for i in range(nZones):
        skimDistance[i, i] = 0.7 * np.min(skimDistance[i, skimDistance[i, :] > 0])
    skimDistance = skimDistance.flatten()

    depotIDs   = list(parcelNodes['id'])

    # ----------------------- Forming spatial clusters of parcels -----------------
    logger.info('Forming spatial clusters of parcels...')

    # A measure of euclidean distance based on the coordinates
    skimEuclidean  = (np.array(list(zonesX.values())).repeat(nZones).reshape(nZones,nZones) -
                      np.array(list(zonesX.values())).repeat(nZones).reshape(nZones,nZones)\
                        .transpose())**2
    skimEuclidean += (np.array(list(zonesY.values())).repeat(nZones).reshape(nZones,nZones) -
                      np.array(list(zonesY.values())).repeat(nZones).reshape(nZones,nZones)\
                        .transpose())**2
    skimEuclidean = skimEuclidean**0.5
    skimEuclidean = skimEuclidean.flatten()
    skimEuclidean /= np.sum(skimEuclidean)

    # To prevent instability related to possible mistakes in skim,
    # use average of skim and euclidean distance (both normalized to a sum of 1)
    skimClustering  = skimDistance.copy()
    skimClustering = skimClustering / np.sum(skimClustering)
    skimClustering += skimEuclidean

    del skimEuclidean

    if cfg['LABEL'] == 'UCC':

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
            print('\tTour type ' + str(i+1) + '...')
            parcelsUCC[i] = cluster_parcels(parcelsUCC[i], maxVehicleLoad, skimClustering)

        # LEVV have smaller capacity
        print('\tTour type 4...')
        parcelsUCC[3] = cluster_parcels(parcelsUCC[3], int(round(maxVehicleLoad/5)), skimClustering)

        # Aggregate parcels based on depot, cluster and destination
        for i in range(4):
            if i <= 1:
                parcelsUCC[i] = pd.pivot_table(parcelsUCC[i],
                                               values=['Parcel_ID'],
                                               index=['DepotNumber', 'Cluster',
                                                      'O_zone', 'D_zone'],
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

    if cfg['LABEL'] != 'UCC':
        # Cluster parcels based on proximity and constrained by vehicle capacity
        parcels = cluster_parcels(parcels, maxVehicleLoad, skimClustering)

        # Aggregate parcels based on depot, cluster and destination
        parcels = pd.pivot_table(parcels,
                                 values=['Parcel_ID'],
                                 index=['DepotNumber', 'Cluster', 'O_zone', 'D_zone'],
                                 aggfunc = {'Parcel_ID': 'count'})
        parcels = parcels.rename(columns={'Parcel_ID':'Parcels'})
        parcels['Depot'  ] = [x[0] for x in parcels.index]
        parcels['Cluster'] = [x[1] for x in parcels.index]
        parcels['Orig'   ] = [x[2] for x in parcels.index]
        parcels['Dest'   ] = [x[3] for x in parcels.index]
        parcels.index = np.arange(len(parcels))

    del skimClustering

    # ----------- Scheduling of trips (UCC scenario) --------------------------
    if cfg['LABEL'] == 'UCC':

        # Depots to households
        print('Starting scheduling procedure for parcels from depots to households...')

        tourType = 0
        deliveries = create_schedules(parcelsUCC[0], dropOffTime, skimTravTime, skimDistance,
                                      parcelNodesCEP, parcelDepTime, tourType)

        # Depots to UCCs
        print('Starting scheduling procedure for parcels from depots to UCC...')

        tourType = 1
        deliveries1 = create_schedules(parcelsUCC[1], dropOffTime, skimTravTime, skimDistance,
                                       parcelNodesCEP, parcelDepTime, tourType)

        # Depots to UCCs (van)
        print('Starting scheduling procedure for parcels from UCCs (by van)...')

        tourType = 2
        deliveries2 = create_schedules(parcelsUCC[2], dropOffTime, skimTravTime, skimDistance,
                                       parcelNodesCEP, parcelDepTime, tourType)

        # Depots to UCCs (LEVV)
        print('Starting scheduling procedure for parcels from UCCs (by LEVV)...')

        tourType = 3
        deliveries3 = create_schedules(parcelsUCC[3], dropOffTime, skimTravTime, skimDistance,
                                       parcelNodesCEP, parcelDepTime, tourType)


        # Combine deliveries of all tour types
        deliveries = pd.concat([deliveries, deliveries1, deliveries2, deliveries3])
        deliveries.index = np.arange(len(deliveries))


    # ----------- Scheduling of trips (REF scenario) ----------------------------
    if cfg['LABEL'] != 'UCC':
        logger.info('Starting scheduling procedure for parcels...')

        tourType = 0

        deliveries = create_schedules(parcels, dropOffTime, skimTravTime, skimDistance,
                                      parcelNodesCEP, parcelDepTime, tourType)


    # ------------------ Export output table to CSV and SHP -------------------
    # Transform to MRDH zone numbers and export
    deliveries['O_zone']  =  [zoneDict[x] for x in deliveries['O_zone']]
    deliveries['D_zone']  =  [zoneDict[x] for x in deliveries['D_zone']]
    deliveries['TripDepTime'] = [round(deliveries['TripDepTime'][i], 3) for i in deliveries.index]
    deliveries['TripEndTime'] = [round(deliveries['TripEndTime'][i], 3) for i in deliveries.index]

    logger.info("Writing scheduled trips to ParcelSchedule.csv")
    filename = "ParcelSchedule.csv" if not hub2hub else "ParcelSchedule_Hub2Hub.csv"
    deliveries.to_csv(join(cfg['OUTDIR'], filename), index=False)

    # ------------------------ Create and export trip matrices ----------------
    if exportTripMatrix:
        logger.info('Generating trip matrix...')
        cols = ['ORIG','DEST', 'N_TOT']
        deliveries['N_TOT'] = 1

        # Gebruik N_TOT om het aantal ritten per HB te bepalen,
        # voor elk logistiek segment, voertuigtype en totaal
        pivotTable = pd.pivot_table(deliveries, values=['N_TOT'], index=['O_zone','D_zone'],
                                    aggfunc=np.sum)
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

        pivotTable.to_csv(join(cfg['OUTDIR'], "tripmatrix_parcels.txt"), index=False, sep='\t')
        logger.info('Trip matrix written to file')

        deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24
        deliveries.loc[deliveries['TripDepTime']>=24,'TripDepTime'] -= 24


def run_unconsolidated_trips(cfg: dict, invZoneDict: dict,
                             skimTravTime: np.array, skimDist_flat: np.array,
                             nSkimZones: int, cepNodeDict: dict) -> dict:
    """_summary_

    :param cfg: _description_
    :type cfg: dict
    :param invZoneDict: _description_
    :type invZoneDict: dict
    :param skimTravTime: _description_
    :type skimTravTime: np.array
    :param skimDist_flat: _description_
    :type skimDist_flat: np.array
    :param nSkimZones: _description_
    :type nSkimZones: int
    :param cepNodeDict: _description_
    :type cepNodeDict: dict
    :return: _description_
    :rtype: dict
    """
    execute(cfg)

    parcels = pd.read_csv(cfg['PARCELS'])

    parcelschedule_hubspoke = pd.read_csv(join(cfg['OUTDIR'], "ParcelSchedule.csv"))
    parcelschedule_hubspoke_delivery = \
        parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Delivery'].copy()
    parcelschedule_hubspoke_pickup = \
        parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Pickup'].copy()


    parcelschedule_hubspoke_delivery['connected_tour'] = ''
    parcelschedule_hubspoke_pickup['connected_tour'] = ''
    delivery_tour_end = \
        parcelschedule_hubspoke_delivery.drop_duplicates(subset = ["Tour_ID"], keep='last').copy()
    pickup_tour_start = \
        parcelschedule_hubspoke_pickup.drop_duplicates(subset = ["Tour_ID"], keep='first').copy()

    kpis = {}
    kpis['conventional_direct_return'] = 0
    if cfg['COMBINE_DELIVERY_PICKUP_TOUR']:
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
                best_tour = possible_pickuptours.iloc[skimTravTime[invZoneDict[endzone]-1,
                                                      possible_pickupzones].argmin()]
                time_direct = skimTravTime[invZoneDict[delivery['O_zone']]-1,
                                           invZoneDict[best_tour['D_zone']]-1]
                time_via_depot = (skimTravTime[invZoneDict[delivery['O_zone']]-1,
                                  invZoneDict[delivery['D_zone']]-1] + \
                                    skimTravTime[invZoneDict[best_tour['O_zone']] - 1,
                                                             invZoneDict[best_tour['D_zone']]-1])
                if time_direct < time_via_depot:
                    pickup_tour_start.loc[best_tour.name, 'connected_tour'] = delivery['Tour_ID']
                    parcelschedule_hubspoke_pickup.loc[best_tour.name, 'connected_tour'] = \
                        delivery['Tour_ID']
                    delivery_tour_end.loc[index, 'connected_tour'] = best_tour['Tour_ID']
                    parcelschedule_hubspoke_delivery.loc[index, 'connected_tour'] = \
                        best_tour['Tour_ID']

                    kpis['conventional_direct_return'] += \
                        round(get_distance(invZoneDict[delivery['O_zone']],
                                           invZoneDict[best_tour['D_zone']],
                                           skimDist_flat,
                                           nSkimZones), 2)
                    # Shouldn't you be substracting the return distance to the depot?

    # Conventional KPI's
    # Parcel scheduling
    distance = 0
    distanceDHL = 0
    distanceDPD= 0
    distanceFedEx= 0
    distanceGLS= 0
    distancePostNL = 0
    distanceUPS= 0
    ParcelSchedules = [parcelschedule_hubspoke_delivery, parcelschedule_hubspoke_pickup]
    for ParcelSchedule in ParcelSchedules:
        for index, delivery in ParcelSchedule.iterrows():
            if delivery['connected_tour'] != '':
                continue
            else:
                orig = invZoneDict[delivery['O_zone']]
                dest = invZoneDict[delivery['D_zone']]
                distance += get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'DHL':
                    distanceDHL +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'DPD':
                    distanceDPD +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'FedEx':
                    distanceFedEx +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'GLS':
                    distanceGLS +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'PostNL':
                    distancePostNL +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'UPS':
                    distanceUPS +=get_distance(orig, dest, skimDist_flat, nSkimZones)

    distances = {
        "total" : distance,
        "DHL": distanceDHL ,
        "DPD": distanceDPD ,
        "FedEx": distanceFedEx ,
        "GLS": distanceGLS ,
        "PostNL": distancePostNL ,
        "UPS": distanceUPS ,
        "totalCheck": distanceDHL + distanceDPD + distanceFedEx + distanceGLS + \
            distancePostNL + distanceUPS
    }
    kpis['conventional_parcels_delivered'] = ParcelSchedules[0]['N_parcels'].sum()
    kpis['conventional_parcels_picked-up'] = ParcelSchedules[1]['N_parcels'].sum()
    kpis['conventional_parcels_total'] = \
        len(np.unique(parcels['Parcel_ID'].append(parcels['Parcel_ID'])))
    ### alleen den haag
    kpis['conventional_distance'] = int(distance + kpis['conventional_direct_return'])
    kpis['conventional_trips'] = sum([len(ParcelSchedule) for ParcelSchedule in ParcelSchedules])
    kpis['conventional_tours'] = sum([len(np.unique(ParcelSchedule['Tour_ID']))
                                      for ParcelSchedule in ParcelSchedules])
    kpis['conventional_distance_avg'] = round(distance / kpis['conventional_parcels_total'], 2)

    CEPflowDeliver = {}
    CEPflowPuP = {}

    for courier,depots in cepNodeDict.items():
        CEPflowDeliver[courier] = {}
        CEPflowPuP[courier] ={}
        Deliver = 0
        PickeUp = 0
        for depot in depots:
            Dep = depot +1 # In the dictionary it uses the Index, not the ID of the depot
            parcelsDelivered = \
                ParcelSchedules[0][ParcelSchedules[0]['Depot_ID']==Dep]['N_parcels'].sum()
            parcelsPickedUp  = \
                ParcelSchedules[1][ParcelSchedules[1]['Depot_ID']==Dep]['N_parcels'].sum()
            CEPflowDeliver[courier][Dep] = parcelsDelivered
            CEPflowPuP[courier][Dep] = parcelsPickedUp
            Deliver += parcelsDelivered
            PickeUp += parcelsPickedUp
        CEPflowDeliver[courier]['Sum'] = Deliver
        CEPflowPuP[courier]['Sum'] =PickeUp

    kpis["deliverFlows"] = CEPflowDeliver
    kpis["pickUpFlows"] = CEPflowPuP
    kpis["distances"] = distances

    return kpis


def run_hub2hub(cfg: dict, invZoneDict: dict,
                skimTravTime: np.array, skimDist_flat: np.array,
                nSkimZones: int, cepNodeDict: dict) -> dict:
    """_summary_

    :param cfg: _description_
    :type cfg: dict
    :param invZoneDict: _description_
    :type invZoneDict: dict
    :param skimTravTime: _description_
    :type skimTravTime: np.array
    :param skimDist_flat: _description_
    :type skimDist_flat: np.array
    :param nSkimZones: _description_
    :type nSkimZones: int
    :param cepNodeDict: _description_
    :type cepNodeDict: dict
    :return: _description_
    :rtype: dict
    """
    execute(cfg)
    execute(cfg, hub2hub=True)

    parcels = pd.read_csv(cfg['PARCELS'])

    parcelschedule_hubspoke = pd.read_csv(join(cfg["OUTDIR"], "ParcelSchedule.csv"))
    parcelschedule_hubspoke_delivery = \
        parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Delivery'].copy()
    parcelschedule_hubspoke_pickup = \
        parcelschedule_hubspoke[parcelschedule_hubspoke['Type'] == 'Pickup'].copy()

    parcelschedule_hubspoke_delivery['connected_tour'] = ''
    parcelschedule_hubspoke_pickup['connected_tour'] = ''
    delivery_tour_end = \
        parcelschedule_hubspoke_delivery.drop_duplicates(subset = ["Tour_ID"], keep='last').copy()
    pickup_tour_start = \
        parcelschedule_hubspoke_pickup.drop_duplicates(subset = ["Tour_ID"], keep='first').copy()

    kpis = {}
    kpis['conventional_direct_return'] = 0
    if cfg['COMBINE_DELIVERY_PICKUP_TOUR']:
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
                best_tour = possible_pickuptours.iloc[skimTravTime[invZoneDict[endzone]-1,
                                                      possible_pickupzones].argmin()]
                time_direct = skimTravTime[invZoneDict[delivery['O_zone']]-1,
                                           invZoneDict[best_tour['D_zone']]-1]
                time_via_depot = (
                    skimTravTime[invZoneDict[delivery['O_zone']]-1,
                                 invZoneDict[delivery['D_zone']]-1] + \
                                    skimTravTime[invZoneDict[best_tour['O_zone']]-1,
                                                 invZoneDict[best_tour['D_zone']]-1]
                )
                if time_direct < time_via_depot:
                    pickup_tour_start.loc[best_tour.name, 'connected_tour'] = delivery['Tour_ID']
                    parcelschedule_hubspoke_pickup.loc[best_tour.name, 'connected_tour'] = \
                        delivery['Tour_ID']
                    delivery_tour_end.loc[index, 'connected_tour'] = best_tour['Tour_ID']
                    parcelschedule_hubspoke_delivery.loc[index, 'connected_tour'] = \
                        best_tour['Tour_ID']

                    kpis['conventional_direct_return'] += round(
                        get_distance(invZoneDict[delivery['O_zone']], invZoneDict[best_tour['D_zone']],
                                     skimDist_flat, nSkimZones), 2)
                    # Shouldn't you be substracting the return distance to the depot?

    # Conventional KPI's
    # Parcel scheduling
    distance = 0
    distanceDHL = 0
    distanceDPD= 0
    distanceFedEx= 0
    distanceGLS= 0
    distancePostNL = 0
    distanceUPS= 0
    ParcelSchedules = [parcelschedule_hubspoke_delivery, parcelschedule_hubspoke_pickup]
    for ParcelSchedule in ParcelSchedules:
        for index, delivery in ParcelSchedule.iterrows():
            if delivery['connected_tour'] != '': continue
            else:
                orig = invZoneDict[delivery['O_zone']]
                dest = invZoneDict[delivery['D_zone']]
                distance += get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'DHL':
                    distanceDHL +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'DPD':
                    distanceDPD +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'FedEx':
                    distanceFedEx +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'GLS':
                    distanceGLS +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'PostNL':
                    distancePostNL +=get_distance(orig, dest, skimDist_flat, nSkimZones)
                if delivery['CEP'] == 'UPS':
                    distanceUPS +=get_distance(orig, dest, skimDist_flat, nSkimZones)
    LastMileDistance = distance
    distances = {
        "total": float(distance),
        "DHL": float(distanceDHL) ,
        "DPD": float(distanceDPD),
        "FedEx": float(distanceFedEx),
        "GLS": float(distanceGLS),
        "PostNL": float(distancePostNL),
        "UPS": float(distanceUPS),
        "totalCheck": float(distanceDHL + distanceDPD + distanceFedEx + distanceGLS + \
            distancePostNL + distanceUPS)
        }
    kpis['conventional_parcels_delivered'] = int(ParcelSchedules[0]['N_parcels'].sum())
    kpis['conventional_parcels_picked-up'] = int(ParcelSchedules[1]['N_parcels'].sum())
    kpis['conventional_parcels_total'] = \
        len(np.unique(parcels['Parcel_ID'].append(parcels['Parcel_ID'])))
    ### alleen den haag
    kpis['conventional_distance'] = int(distance + kpis['conventional_direct_return'])
    kpis['conventional_trips'] = sum([len(ParcelSchedule) for ParcelSchedule in ParcelSchedules])
    kpis['conventional_tours'] = sum([len(np.unique(ParcelSchedule['Tour_ID']))
                                      for ParcelSchedule in ParcelSchedules])
    kpis['conventional_distance_avg'] = round(distance / kpis['conventional_parcels_total'],2)

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
            parcelsDelivered = \
                ParcelSchedules[0][ParcelSchedules[0]['Depot_ID']==Dep]['N_parcels'].sum()
            parcelsPickedUp  = \
                ParcelSchedules[1][ParcelSchedules[1]['Depot_ID']==Dep]['N_parcels'].sum()
            CEPflowDeliver[courier][int(Dep)] = int(parcelsDelivered)
            CEPflowPuP[courier][int(Dep)] = int(parcelsPickedUp)
            Deliver += parcelsDelivered
            PickeUp += parcelsPickedUp
        CEPflowDeliver[courier]['Sum'] = int(Deliver)
        CEPflowPuP[courier]['Sum'] = int(PickeUp )

    kpis["deliverFlows"] = CEPflowDeliver
    kpis["pickUpFlows"] = CEPflowPuP
    kpis["distances"] = distances

    Cons_distance = 0
    Cons_distanceDHL = 0
    Cons_distanceDPD= 0
    Cons_distanceFedEx= 0
    Cons_distanceGLS= 0
    Cons_distancePostNL = 0
    Cons_distanceUPS= 0

    parcelschedule_hub2Hub = pd.read_csv(join(cfg["OUTDIR"], "ParcelSchedule_Hub2Hub.csv"))
    Condistance=0
    # print(parcelschedule_hub2Hub)
    for index, delivery in parcelschedule_hub2Hub.iterrows():
        orig = invZoneDict[delivery['O_zone']]
        dest = invZoneDict[delivery['D_zone']]
        Condistance += get_distance(orig, dest, skimDist_flat, nSkimZones)
        if delivery['CEP'] == 'DHL':
            Cons_distanceDHL += get_distance(orig, dest, skimDist_flat, nSkimZones)
        if delivery['CEP'] == 'DPD':
            Cons_distanceDPD += get_distance(orig, dest, skimDist_flat, nSkimZones)
        if delivery['CEP'] == 'FedEx':
            Cons_distanceFedEx += get_distance(orig, dest, skimDist_flat, nSkimZones)
        if delivery['CEP'] == 'GLS':
            Cons_distanceGLS += get_distance(orig, dest, skimDist_flat, nSkimZones)
        if delivery['CEP'] == 'PostNL':
            Cons_distancePostNL += get_distance(orig, dest, skimDist_flat, nSkimZones)
        if delivery['CEP'] == 'UPS':
            Cons_distanceUPS += get_distance(orig, dest, skimDist_flat, nSkimZones)
        Cons_distance +=Condistance

    ConsolidatedPerCEP = {
        "total": float(Condistance),
        "DHL": float(Cons_distanceDHL),
        "DPD": float(Cons_distanceDPD),
        "FedEx": float(Cons_distanceFedEx),
        "GLS": float(Cons_distanceGLS),
        "PostNL": float(Cons_distancePostNL),
        "UPS": float(Cons_distanceUPS),
    }

    kpis['ConsolidatedPerCEP'] = ConsolidatedPerCEP

    # This is missing some consolidated trips
    kpis['TotalDistances']  = {
        "total" : Condistance + LastMileDistance,
        "DHL": Cons_distanceDHL +distanceDHL,
        "DPD": Cons_distanceDPD +distanceDPD,
        "FedEx": Cons_distanceFedEx +distanceFedEx,
        "GLS": Cons_distanceGLS +distanceGLS,
        "PostNL": Cons_distancePostNL +distancePostNL,
        "UPS": Cons_distanceUPS +distanceUPS,
    }

    return kpis


def run_model(cfg: dict) -> None:
    """_summary_

    :param cfg: _description_
    :type cfg: dict
    """
    start_time = time()

    zones = read_shape(cfg['ZONES'])
    zones.index = zones['AREANR']
    nZones = len(zones)

    skims = {'time': {}, 'dist': {}, }
    skims['time']['path'] = cfg['SKIMTIME']
    skims['dist']['path'] = cfg['SKIMDISTANCE']
    for skim in skims:
        skims[skim] = read_mtx(skims[skim]['path'])
        nSkimZones = int(len(skims[skim])**0.5)
        skims[skim] = skims[skim].reshape((nSkimZones, nSkimZones))
        if skim == 'time':
             # data deficiency
            skims[skim][6483] = skims[skim][:, 6483] = 5000
        # add traveltimes to internal zonal trips
        for i in range(nSkimZones):
            skims[skim][i, i] = 0.7 * np.min(skims[skim][i, skims[skim][i, :] > 0])
    skimTravTime = skims['time']
    skimDist = skims['dist']
    skimDist_flat = skimDist.flatten()

    zoneDict  = dict(np.transpose(np.vstack( (np.arange(1,nZones+1), zones['AREANR']) )))
    zoneDict  = {int(a): int(b) for a, b in zoneDict.items()}
    invZoneDict = dict((v, k) for k, v in zoneDict.items())

    segs = pd.read_csv(cfg['SEGS'])
    segs.index = segs['zone']
    # Take only segs into account for which zonal data is known as well
    segs = segs[segs['zone'].isin(zones['AREANR'])]

    parcelNodesPath = cfg['PARCELNODES']
    parcelNodes = read_shape(parcelNodesPath, return_geometry=False)
    parcelNodes.index = parcelNodes['id'].astype(int)
    parcelNodes = parcelNodes.sort_index()

    for node in parcelNodes['id']:
        parcelNodes.loc[node, 'SKIMNR'] = int(invZoneDict[parcelNodes.at[int(node), 'AREANR']])
    parcelNodes['SKIMNR'] = parcelNodes['SKIMNR'].astype(int)

    cepList = np.unique(parcelNodes['CEP'])
    cepNodes = [np.where(parcelNodes['CEP']==str(cep))[0] for cep in cepList]

    cepNodeDict = {}
    cepZoneDict = {}
    cepSkimDict = {}
    for cep in cepList:
        cepZoneDict[cep] = parcelNodes[parcelNodes['CEP'] == cep]['AREANR'].astype(int).tolist()
        cepSkimDict[cep] = parcelNodes[parcelNodes['CEP'] == cep]['SKIMNR'].astype(int).tolist()
    for cepNo in range(len(cepList)):
        cepNodeDict[cepList[cepNo]] = cepNodes[cepNo]

    kpis = {}
    if not cfg['CONSOLIDATED_TRIPS']:
        kpis = run_unconsolidated_trips(cfg, invZoneDict, skimTravTime, skimDist_flat,
                                        nSkimZones, cepNodeDict)
    else:
        kpis = run_hub2hub(cfg, invZoneDict, skimTravTime, skimDist_flat,
                           nSkimZones, cepNodeDict)

    with open(join(cfg["OUTDIR"], "kpis.json"), "w", encoding="utf-8") as fp:
        dump(kpis, fp, indent=4)

    # Finalize
    totaltime = round(time() - start_time, 2)
    logger.info("Total runtime: %s seconds", totaltime)
