"""Parcel Tour Formation utilities
"""
import array
from logging import getLogger
from os.path import getsize, dirname
from posixpath import splitext
from zipfile import ZipFile

import pandas as pd
import numpy as np
import shapefile as shp


logger = getLogger("parceltourformation.utils")


def get_traveltime(orig, dest, skim, n_zones, time_fac):
    """Obtain the travel time [h] for orig to a destination zone.

    :param orig: _description_
    :type orig: _type_
    :param dest: _description_
    :type dest: _type_
    :param skim: _description_
    :type skim: _type_
    :param n_zones: _description_
    :type n_zones: _type_
    :param time_fac: _description_
    :type time_fac: _type_
    :return: _description_
    :rtype: _type_
    """

    return skim[(orig-1)*n_zones + (dest-1)] * time_fac / 3600


def get_distance(orig, dest, skim, n_zones):
    """Obtain the distance [km] for orig to a destination zone.

    :param orig: _description_
    :type orig: _type_
    :param dest: _description_
    :type dest: _type_
    :param skim: _description_
    :type skim: _type_
    :param n_zones: _description_
    :type n_zones: _type_
    :return: _description_
    :rtype: _type_
    """

    return skim[(orig-1)*n_zones + (dest-1)] / 1000


def read_mtx(mtxfile):
    """Read a binary mtx-file (skimTijd and skimAfstand)

    :param mtxfile: _description_
    :type mtxfile: _type_
    :return: _description_
    :rtype: _type_
    """

    mtx_data = array.array('i')
    with open(mtxfile, 'rb') as fp_mtx:
        mtx_data.fromfile(fp_mtx, getsize(mtxfile) // mtx_data.itemsize)

    # The number of zones is in the first byte
    mtx_data = np.array(mtx_data, dtype=int)[1:]

    return mtx_data


def read_shape(fpath, encoding='latin1', return_geometry=False):
    '''
    Read the shapefile with zones (using pyshp --> import shapefile as shp)
    '''
    # Load the shape
    with ZipFile(fpath) as z:
        z.extractall(path=dirname(fpath))

    shape_path = splitext(fpath)[0] + '.shp'
    sf_reader = shp.Reader(shape_path, encoding=encoding)
    records = sf_reader.records()
    if return_geometry:
        geometry = sf_reader.__geo_interface__
        geometry = geometry['features']
        geometry = [geometry[i]['geometry'] for i in range(len(geometry))]
    fields = sf_reader.fields
    sf_reader.close()

    # Get information on the fields in the DBF
    columns  = [x[0] for x in fields[1:]]
    col_types = [x[1:] for x in fields[1:]]
    n_records = len(records)

    # Put all the data records into a NumPy array (much faster than Pandas DataFrame)
    shape = np.zeros((n_records,len(columns)), dtype=object)
    for i in range(n_records):
        shape[i,:] = records[i][0:]

    # Then put this into a Pandas DataFrame with the right headers and data types
    shape = pd.DataFrame(shape, columns=columns)
    for i_c, column in enumerate(columns):
        if col_types[i_c][0] == 'C':
            shape[column] = shape[column].astype(str)
        else:
            shape.loc[pd.isna(shape[column]), column] = -99999
            if col_types[i_c][-1] > 0:
                shape[column] = shape[column].astype(float)
            else:
                shape[column] = shape[column].astype(int)

    if return_geometry:
        return (shape, geometry)

    return shape


def create_geojson(output_path, dataframe, origin_x, origin_y, destination_x, destination_y):
    """Creates GEO JSON file

    :param output_path: The output path to create the file
    :type output_path: str
    :param dataframe: The data
    :type dataframe: pd.Dataframe
    :param origin_x: X origin index
    :type origin_x: _type_
    :param origin_y: Y origin index
    :type origin_y: _type_
    :param destination_x: X destination index
    :type destination_x: _type_
    :param destination_y: Y destination index
    :type destination_y: _type_
    """
    a_x = np.array(dataframe[origin_x], dtype=str)
    a_y = np.array(dataframe[origin_y], dtype=str)
    b_x = np.array(dataframe[destination_x], dtype=str)
    b_y = np.array(dataframe[destination_y], dtype=str)
    n_trips = len(dataframe.index)

    with open(output_path, 'w') as geo_file:        # FIXME: no encoding
        geo_file.write('{\n' + '"type": "FeatureCollection",\n' + '"features": [\n')
        for i in range(n_trips-1):
            output_str = ""
            output_str = output_str + '{ "type": "Feature", "properties": '
            output_str = output_str + str(dataframe.loc[i, :].to_dict()).replace("'", '"')
            output_str = output_str + ', "geometry": { "type": "LineString", "coordinates": [ [ '
            output_str = output_str + a_x[i] + ', ' + a_y[i] + ' ], [ '
            output_str = output_str + b_x[i] + ', ' + b_y[i] + ' ] ] } },\n'
            geo_file.write(output_str)

        # Bij de laatste feature moet er geen komma aan het einde
        i += 1
        output_str = ""
        output_str = output_str + '{ "type": "Feature", "properties": '
        output_str = output_str + str(dataframe.loc[i, :].to_dict()).replace("'", '"')
        output_str = output_str + ', "geometry": { "type": "LineString", "coordinates": [ [ '
        output_str = output_str + a_x[i] + ', ' + a_y[i] + ' ], [ '
        output_str = output_str + b_x[i] + ', ' + b_y[i] + ' ] ] } }\n'
        geo_file.write(output_str)
        geo_file.write(']\n')
        geo_file.write('}')


def cluster_parcels(parcels, maxVehicleLoad, skimDistance):
    """Assign parcels to clusters based on spatial proximity with cluster size constraints.
    The cluster variable is added as extra column to the DataFrame.

    :param parcels: _description_
    :type parcels: _type_
    :param maxVehicleLoad: _description_
    :type maxVehicleLoad: _type_
    :param skimDistance: _description_
    :type skimDistance: _type_
    :return: _description_
    :rtype: _type_
    """
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

    # TODO: allow multiple vehicles?
    parcels['VEHTYPE'] = 'Van' # Making it such that there is one type of vans (TO CHANGE)

    counts = pd.pivot_table(parcels, values=['VEHTYPE'], index=['DepotNumber','D_zone'],
                            aggfunc=len)

    whereLargeCluster = list(counts.index[np.where(counts>=maxVehicleLoad)[0]])
    for x in whereLargeCluster:
        depotNumber = x[0]
        destZone    = x[1]

        indices = np.where((parcels['DepotNumber']==depotNumber) & \
                           (parcels['D_zone']==destZone))[0]

        for _ in range(int(np.floor(len(indices)/maxVehicleLoad))):
            parcels.loc[indices[:maxVehicleLoad], 'Cluster'] = firstClusterID
            indices = indices[maxVehicleLoad:]

            firstClusterID += 1
            nParcelsAssigned += maxVehicleLoad

            print('\t' + str(int(round((nParcelsAssigned/nParcels)*100,0))) + '%', end='\r')

    # For each depot, cluster remaining parcels into batches of {maxVehicleLoad} parcels
    for depotNumber in depotNumbers:
        # Select parcels of the depot that are not assigned a cluster yet
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

            for _ in range(nTours):
                # Select the first parcel for the new cluster that is now initialized
                yetAssigned    = (clusters!=-1)
                notYetAssigned = np.where(~yetAssigned)[0]
                firstParcelIndex = notYetAssigned[0]
                clusters[firstParcelIndex] = firstClusterID

                # Find the nearest {maxVehicleLoad-1} parcels to this first parcel that are not
                # in a cluster yet
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
            nParcelsAssigned += len(parcelsToFit)

            print('\t' + str(int(round((nParcelsAssigned/nParcels)*100,0))) + '%', end='\r')

    parcels['Cluster'] = parcels['Cluster'].astype(int)

    return parcels


def create_schedules(parcelsAgg, dropOffTime, skimTravTime, skimDistance,
                     parcelNodesCEP, parcelDepTime, tourType):
    """_summary_

    :param parcelsAgg: _description_
    :type parcelsAgg: _type_
    :param dropOffTime: _description_
    :type dropOffTime: _type_
    :param skimTravTime: _description_
    :type skimTravTime: _type_
    :param skimDistance: _description_
    :type skimDistance: _type_
    :param parcelNodesCEP: _description_
    :type parcelNodesCEP: _type_
    :param parcelDepTime: _description_
    :type parcelDepTime: _type_
    :param tourType: _description_
    :type tourType: _type_
    :return: _description_
    :rtype: _type_
    """
    # Create the parcel schedules and store them in a DataFrame
    nZones = int(len(skimTravTime)**0.5)
    depots = np.unique(parcelsAgg['Depot'])
    nDepots = len(depots)

    print('\t0%', end='\r')

    tours            = {}
    parcelsDelivered = {}
    departureTimes   = {}
    depotCount = 0
    nTrips     = 0

    for depot in np.unique(parcelsAgg['Depot']):
        depotParcels = parcelsAgg[parcelsAgg['Depot']==depot]

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

            # Shuffle the order of tour locations and accept the shuffle if it reduces the tour
            # distance
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
                            swappedTourDist = \
                                np.sum(skimDistance[swappedTour[:-1] * nZones + swappedTour[1:]])

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
                # Depot to HH (0) or UCC (1), UCC to HH by van (2)/LEVV (3)
                deliveries[tripcount, 0] = tourType
                if tourType <= 1:
                    # Name of the couriers
                    deliveries[tripcount, 1] = parcelNodesCEP[depot]
                else:
                    # Name of the couriers
                    deliveries[tripcount, 1] = 'ConsolidatedUCC'
                if depot < 1000:
                    # Depot_ID
                    deliveries[tripcount, 2] = depot
                    # Tour_ID
                    deliveries[tripcount, 3] = f'{depot}_{tour}'
                    # Trip_ID
                    deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'
                    # Unique ID under consideration of tour type
                    deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'
                # EDIT 6-10: SHADOW NODES - start
                if depot >= 1000:
                    depot = depot - 1000
                    # Depot_ID
                    deliveries[tripcount, 2] = depot
                    # Tour_ID
                    deliveries[tripcount, 3] = f'{depot}_{tour}'
                    # Trip_ID
                    deliveries[tripcount, 4] = f'{depot}_{tour}_{trip}'
                    # Unique ID under consideration of tour type
                    deliveries[tripcount, 5] = f'{depot}_{tour}_{trip}_{tourType}'
                    depot = depot + 1000
                # EDIT 6-10: SHADOW NODES - end
                # Origin
                deliveries[tripcount, 6] = orig
                # Destination
                deliveries[tripcount, 7] = dest
                # Number of parcels
                deliveries[tripcount, 8] = parcelsDelivered[depot][tour][trip]
                # Travel time in hrs
                deliveries[tripcount, 9] = skimTravTime[orig * nZones + dest] / 3600
                # Departure of tour from depot
                deliveries[tripcount,10] = departureTimes[depot][tour][0]
                # Departure time of trip
                deliveries[tripcount,11] = departureTimes[depot][tour][trip]
                # End of trip/start of next trip if there is another one
                deliveries[tripcount,12] = 0.0
                # EDIT 6-10: SHADOW NODES - start
                deliveries[tripcount,13] = 'Delivery'
                if depot >= 1000:
                    deliveries[tripcount,13] = 'Pickup'
                # EDIT 6-10: SHADOW NODES - end
                deliveries[tripcount,14] = get_distance(orig, dest, skimDist_flat, nZones)
                tripcount += 1

    # Place in DataFrame with the right data type per column
    deliveries = pd.DataFrame(deliveries, columns=deliveriesCols)
    dtypes =  {'TourType':int,      'CEP':str,          'Depot_ID':int,      'Tour_ID':str,
               'Trip_ID':str,       'Unique_ID':str,    'O_zone':int,        'D_zone':int,
               'N_parcels':int,     'Traveltime':float, 'TourDepTime':float, 'TripDepTime':float,
               'TripEndTime':float, 'Type':str,"TourDist":float}
    for col in range(len(deliveriesCols)):
        deliveries[deliveriesCols[col]] = \
            deliveries[deliveriesCols[col]].astype(dtypes[deliveriesCols[col]])

    vehTypes  = ['Van',   'Van',   'Van', 'LEVV']
    origTypes = ['Depot', 'Depot', 'UCC', 'UCC']
    destTypes = ['HH',    'UCC',   'HH',  'HH']

    deliveries['VehType' ] = vehTypes[tourType]
    deliveries['OrigType'] = origTypes[tourType]
    deliveries['DestType'] = destTypes[tourType]

    return deliveries
