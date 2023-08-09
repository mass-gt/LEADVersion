"""Processing module
"""

from os.path import join

import pandas as pd


OUT_FN = "input-copert.xlsx"


def run_model(cfg: dict) -> None:
    """_summary_

    :param cfg: the configuration dictionary
    :type cfg: dict
    """

    parcel_trips = pd.read_csv(cfg['parcel_activity'] )

    mean_activity = parcel_trips.groupby(['VehType'])["TourDist"].sum()
    mean_activity = mean_activity.reset_index()
    mean_activity.columns = ["Category", cfg['Year']]

    # Get peak and off peak
    activity_peak = parcel_trips[
        ((parcel_trips["TourDepTime"] > cfg["PeakHourMorningStart"] ) & \
         (parcel_trips["TourDepTime"] < cfg["PeakHourMorningFinish"] )) | \
        ((parcel_trips["TourDepTime"] > cfg["PeakHourAfternoonStart"] ) & \
         (parcel_trips["TourDepTime"] < cfg["PeakHourAfternoonFinish"] ))
    ]
    peak_activity = activity_peak.groupby(['VehType'])["TourDist"].sum()
    peak_activity = peak_activity.reset_index()
    peak_activity.columns = ["Category", "Peak" ]

    # Get vehicle types
    vehicle_kms = {}
    vehicle_peak = {}

    for key in cfg["VehicleOffPeakSpeed"].keys():
        vehicle_kms[key] = 0
        vehicle_peak[key] = 0

    # Convert vehicle use from MASS GT into COPERT vehicles
    for key1 in cfg["VehicleType"].keys():
        for key2 in cfg["VehicleType"][key1].keys():
            kms_tot = mean_activity.loc[mean_activity['Category'] == key1]
            kms_peak = peak_activity.loc[peak_activity['Category'] == key1]
            if kms_tot.empty:
                kms_tot = 0
            else:
                kms_tot = kms_tot.iloc[0][cfg["Year"]]
            if kms_peak.empty:
                kms_peak = 0
            else:
                kms_peak = kms_peak.iloc[0]["Peak"]

            vehicle_kms[key2] += kms_tot * cfg["VehicleType"][key1][key2]
            vehicle_kms[key2] = round(vehicle_kms[key2], 2)
            vehicle_peak[key2] += kms_peak * cfg["VehicleType"][key1][key2]
            vehicle_peak[key2] = round(vehicle_peak[key2], 2)

    # Generate peak
    vehicle_peak_perc = {}
    vehicle_off_peak_perc ={}
    for key, value in vehicle_kms.items():
        per = round(vehicle_peak[key] / (value + 0.0000001) * 1, 2)
        vehicle_peak_perc[key] = per
        vehicle_off_peak_perc[key] = 1 - per

    # COPERT setup
    category = []
    fuel = []
    segment = []
    euro_standard = []
    activity = []
    urban_off_peak_speed = []
    urban_peak_speed = []
    urban_off_peak_share = []
    urban_peak_share = []

    for key, value in vehicle_kms.items():
        category.append(cfg["VehicleCat"][key])
        fuel.append(cfg["VehicleFuel"][key])
        segment.append(cfg["VehicleSegment"][key])
        if  key == "Bike":
            euro_standard.append("0")
        elif key == "Hybrid":
            euro_standard.append(str(cfg["VehicleEuroStand"][key][0]))
        elif key == "Electric":
            euro_standard.append("0")
        else:
            euro_standard.append(cfg["VehicleEuroStand"][key])
        activity.append(value)
        urban_off_peak_speed.append(cfg["VehicleOffPeakSpeed"][key])
        urban_peak_speed.append(cfg["VehiclePeakSpeed"][key])
        urban_off_peak_share.append(vehicle_off_peak_perc[key])
        urban_peak_share.append(vehicle_peak_perc[key])

    df_activity = pd.DataFrame(columns=['Category', 'Fuel', 'Segment', 'Euro Standard'])
    df_activity['Category'] = category
    df_activity['Fuel'] = fuel
    df_activity['Segment'] = segment
    df_activity['Euro Standard'] = euro_standard
    df_activity[cfg["Year"]] = activity

    df_mean_activity = df_activity[df_activity["Category"] != 0]

    df_stock = df_activity
    df_stock[cfg["Year"]] = 1
    df_stock = df_stock[df_stock["Category"] != 0]

    df_urban_off_peak_speed = df_activity
    df_urban_off_peak_speed[cfg["Year"]] = urban_off_peak_speed
    df_urban_off_peak_speed = df_urban_off_peak_speed[df_urban_off_peak_speed["Category"] != 0]

    df_urban_peak_speed = df_activity
    df_urban_peak_speed[cfg["Year"]] = urban_peak_speed
    df_urban_peak_speed = df_urban_peak_speed[df_urban_peak_speed["Category"] != 0]

    df_urban_off_peak_share = df_activity
    df_urban_off_peak_share[cfg["Year"]] = urban_off_peak_share
    df_urban_off_peak_share = df_urban_off_peak_share[df_urban_off_peak_share["Category"] != 0]

    df_urban_peak_share = df_activity
    df_urban_peak_share[cfg["Year"]] = urban_peak_share
    df_urban_peak_share = df_urban_peak_share[df_urban_peak_share["Category"] != 0]

    weather = pd.read_csv(cfg['weather'])
    df_min_temperature = weather[["Month", "Min_Temp"]].rename(columns={"Min_Temp": cfg["Year"]})
    df_max_temperature = weather[["Month", "Max_Temp"]].rename(columns={"Max_Temp": cfg["Year"]})
    df_humidity = weather[["Month", "Humidity"]].rename(columns={"Humidity": cfg["Year"]})
    df_humidity[cfg["Year"]] = df_humidity[cfg["Year"]] / 100.

    sheet_names = ["STOCK", "MEAN_ACTIVITY",
                   "URBAN_OFF_PEAK_SPEED", "URBAN_PEAK_SPEED",
                   "URBAN_OFF_PEAK_SHARE", "URBAN_PEAK_SHARE",
                   "MIN_TEMPERATURE", "MAX_TEMPERATURE", "HUMIDITY"]
    sheet_vals = ["[n]", "[km]",
                  "[km/h]", "[km/h]",
                  "[%]", "[%]",
                  "[℃]", "[℃]", "[%]"]
    df_sheets =  pd.DataFrame(list(zip(sheet_names, sheet_vals)),
                           columns =['SHEET_NAME', 'Unit'])
    sheet_hyper = []
    for i in sheet_names:
        string = OUT_FN +"#"+i+"!A1"
        sheet_hyper.append ( "=HYPERLINK("+f'"{string}"'+","+f'"{i}"'+")")
    df_sheets = pd.DataFrame(list(zip(sheet_hyper, sheet_vals)),
                             columns =['SHEET_NAME', 'Unit'])

    with pd.ExcelWriter(join(cfg['outdir'], OUT_FN), engine='xlsxwriter') as writer:    # pylint: disable=abstract-class-instantiated
        df_sheets.to_excel(writer, sheet_name='SHEETS', index=False)
        df_stock.to_excel(writer, sheet_name='STOCK', index=False)
        df_mean_activity.to_excel(writer, sheet_name='MEAN_ACTIVITY', index=False)
        df_urban_off_peak_speed.to_excel(writer, sheet_name='URBAN_OFF_PEAK_SPEED', index=False)
        df_urban_peak_speed.to_excel(writer, sheet_name='URBAN_PEAK_SPEED', index=False)
        df_urban_off_peak_share.to_excel(writer, sheet_name='URBAN_OFF_PEAK_SHARE', index=False)
        df_urban_peak_share.to_excel(writer, sheet_name='URBAN_PEAK_SHARE', index=False)
        df_min_temperature.to_excel(writer, sheet_name='MIN_TEMPERATURE', index=False)
        df_max_temperature.to_excel(writer, sheet_name='MAX_TEMPERATURE', index=False)
        df_humidity.to_excel(writer, sheet_name='HUMIDITY', index=False)
