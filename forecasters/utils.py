"""
@Author - Ayan Mukhopadhyay
Utility methods for forecasting models
"""

import pandas as pd
import numpy as np
from math import ceil, floor
from datetime import timedelta
import pickle
from user_defined_helpers import get_traffic, get_weather

regression_columns_count = ['cluster_label']
regression_columns_time = ['time_bet', 'cluster_label', 'time_of_day', 'weekday', 'past_incidents']
count_models = ['Simple_Regression','Poisson_Regression', 'Negative_Binomial_Regression',
                'Zero_Inflated_Poisson_Regression']


def create_regression_df(df, metadata, model_name):
    """
    Creates a dataframe for the regression model from standard data.

    df: standard dataframe. Must contain spatial id and time of incidents
    model_name: one of the forecasting models
    features: If None, default features are created, else metadata['features'] are used.
    """
    max_cluster_label = max(df.cluster_label)
    rows = []

    # for count based models

    if model_name in count_models:
        windows = ceil(int((metadata['end_time_test'] - metadata['start_time_train']).total_seconds()) /
                       metadata['window_size'])

        temp_start = metadata['start_time_train']

        # iterate through windows and unit_nums
        for i in range(windows):
            temp_end = temp_start + timedelta(seconds=metadata['window_size'])
            for unit_num in metadata['units']:
                # initialize row with count for train, segment_id and cluster label  #smv: what is train?
                temp_row = [((df[metadata['unit_name']] == unit_num) & (df.time >= temp_start) & (df.time < temp_end)).sum(),
                            unit_num, df.loc[df.unit_segment_id == unit_num].iloc[0]['cluster_label']]
                # add features
                temp_row.extend(get_features(df, unit_num, coord=metadata['location_map'][unit_num],
                                             time=temp_start, window_size=metadata['window_size'],
                                             features=metadata['features'],metadata=metadata))
                rows.append(temp_row)
            temp_start += timedelta(seconds=metadata['window_size'])

        # create dataframe -- start time is included as it is useful to report it while predicting. No need in training.
        col = ['count', metadata['unit_name'], 'cluster_label', 'start_time']
        col.extend(metadata['features'])

        df_regression = pd.DataFrame(rows, columns=col)

        # create categorical columns for specified data types
        for col in metadata['cat_features']:
            # create dummy dataframe
            dummy_temp = pd.get_dummies(df_regression[col], prefix='cat_'+col)
            # merge with original dataframe
            df_regression = pd.concat([df_regression, dummy_temp], axis=1)
            # drop the categorical column
            df_regression.drop([col], axis=1, inplace=True)

        # split into train and predict data sets
        df_train = df_regression.loc[(df_regression['start_time']) < metadata['start_time_test']]
        df_predict = df_regression.loc[(df_regression['start_time']) >= metadata['start_time_test']]
        return {'train': df_train, 'predict':df_predict}

    # for time based models
    elif model_name == 'Survival_Regression':
        # create empty dataframe with required columns
        cols = df.columns.tolist()
        cols.append('time_bet')
        df_temp = pd.DataFrame(columns=cols)

        # get arrival data for each unit
        for unit_num in metadata['units']:
            df_unit_num = df.loc[df.unit_segment_id == unit_num]
            df_unit_num.sort_values('time')

            # get time differences and adjust 1st incident with base time
            df_unit_num['time_bet'] = df.time.diff()                                                #smv: check the dimension
            df_unit_num.iloc[0]['time_bet'] = df.iloc[0]['time'] - metadata['start_time_train']
            df_temp = pd.concat([df_temp, df_unit_num], ignore_index=True)                          #smv: check this

        # get features for each row
        for i in range(len(df_temp)):
            unit_num = df_temp.iloc[i][metadata['unit_name']]
            f = get_features(df_temp, unit_num, metadata['location_map'][unit_num], df_temp.iloc[i]['time'],
                             metadata['window_size'],metadata=metadata)

            temp_row = [df_temp.iloc[i]['time_bet'], df_temp.iloc[i]['cluster_label']]
            temp_row.extend(f)

        # create regression dataframe
        df_converted = pd.DataFrame(rows, columns=regression_columns_time)
        return df_converted


    # TODO: RECTIFY THIS --> WHY SEND METADATA BACK??
    return df, metadata


def create_regression_df_test(df, metadata):
    """
    Only creates test df_converted for count based models. For time based models, test points are generated on the fly
    CAVEAT -- currently uses the incident df to calculate past incidents *** Should be replaced ***
    @param df: raw dataframe of incidents
    @param metadata: metadata with start and end dates, spatial unit etc. See github documentation for details
    @return: dataframe in the specific regression format.
    """
    rows = []
    windows = ceil(int((metadata['end_time_test'] - metadata['start_time_test']).total_seconds()) / metadata['window_size'])
    temp_start = metadata['start_time_test']

    # iterate through windows and unit_nums
    for i in range(windows):
        for unit_num in metadata['units']:
            temp_row = [metadata['cluster_label'][unit_num]]
            # add features
            temp_row.extend(get_features(df, unit_num, coord=metadata['location_map'][unit_num],
                                         time=temp_start, window_size=metadata['window_size'],
                                         features=metadata['features'],metadata=metadata))
            temp_row.extend([temp_start, unit_num])
            rows.append(temp_row)

        temp_start += timedelta(seconds=metadata['window_size'])

    # adjust column names -- drop count and add intercept, start_time and unit_num
    column_names = regression_columns_count
    column_names.extend(metadata['features'])
    column_names.extend(['start_time', metadata['unit_name']])
    # create and return dataframe
    df = pd.DataFrame(rows, columns=column_names)
    df['Intercept'] = df.apply(lambda x: 1, axis=1)
    return df


def get_temporal_features(curr_time, feature_name):
    """
    Creates temporal features for a given timestamp
    @param curr_time: timestamp to get temporal features
    @param feature_name: list of feature names
    @return: list of features at given time
    """
    # time of day, weekday
    f = []
    if 'weekday' in feature_name:
        weekday = 1 if curr_time.isoweekday() in range(1, 6) else 0
        f.append(weekday)
    if 'time_of_day' in feature_name:
        f.append(int(floor(curr_time.hour/4)))

    return f


def get_features(df, unit_num, coord, time, window_size, features=None,metadata=None):
    """
    Creates a set of spatio-temporal features for a given space-time unit
    @param df: historical incidents
    @param unit_num: current unit_num
    @param coord: coordinate to get features like weather
    @param time: time of the incident
    @param window_size: time-frame size for discrete space models
    @param features: list of features
    @return:
    """
    # edit feature names in the order they are created, not in the order supplied
    f = []
    for f_name in features:
        if f_name == "time_of_day":
            f.append(int(floor(time.hour / (window_size/3600) )))             
        elif f_name == "weekday":                           
            weekday = 1 if time.isoweekday() in range(1, 6) else 0
            f.append(weekday)
        elif f_name == 'rain':                              
            f.extend(get_weather(metadata,coord, time))
        elif f_name == 'traffic':
            f.extend(get_traffic(metadata,coord, time))
        elif f_name == 'past_incidents':                    
            f.append(((df.unit_segment_id == unit_num) & (df.time > time - timedelta(seconds=window_size))
                         & (df.time <= time)).sum())
        else:
            print("{} is not a valid feature. It is being ignored".format(f_name))
    return f
  

def update_meta_categorical(met_features, df_features, cat_col):
    """
    updates feature set with transformed names for categorical features and deletes old features
    @param met_features: features from metadata
    @param df_features: features from the transformed dataframe
    @param cat_col: names of categorical columns
    @return:
    """
    for f in cat_col:
        f_cat = []
        for i in df_features:
            if "cat_" + f + '_' in i:
                f_cat.append(i)
        met_features.remove(f)
        met_features.extend(f_cat)
    return met_features

