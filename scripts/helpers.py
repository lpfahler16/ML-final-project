import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

df_2018 = pd.read_csv('../data/2018.csv')
df_2017 = pd.read_csv('../data/2017.csv')
df_2016 = pd.read_csv('../data/2016.csv')
df_2015 = pd.read_csv('../data/2015.csv')
df_2014 = pd.read_csv('../data/2014.csv')
df_2013 = pd.read_csv('../data/2013.csv')
df_2012 = pd.read_csv('../data/2012.csv')
df_2011 = pd.read_csv('../data/2011.csv')
df_2010 = pd.read_csv('../data/2010.csv')


def get_classifier_convert(df, nn=False):
    conditions = [
        (df['fourth_down_converted'] == 1.0),
        (df['fourth_down_failed'] == 1.0),
        (df['field_goal_attempt'] == 1.0),
        (df['punt_attempt'] == 1.0)
    ]

    if nn:
        results = [0, 1, 2, 3]
    else:
        results = ['CONVERTED', 'FAILED', 'FIELD_GOAL', 'PUNT']

    y = np.select(conditions, results)
    return y


def get_classifier_attempt(df, nn=False):
    conditions = [
        (df['fourth_down_converted'] == 1.0) | (
            df['fourth_down_failed'] == 1.0),
        (df['field_goal_attempt'] == 1.0),
        (df['punt_attempt'] == 1.0)
    ]

    if nn:
        results = [0, 1, 2]
    else:
        results = ['ATTEMPTED', 'FIELD_GOAL', 'PUNT']

    y = np.select(conditions, results)
    return y


def ready_data_convert(df, nn=False):
    df = df.dropna()
    y = get_classifier_convert(df, nn)
    df = df.drop(columns=['posteam', 'fourth_down_converted', 'fourth_down_failed',
                 'field_goal_attempt', 'punt_attempt', 'game_date', 'down'])
    return df, y


def ready_data_attempt(df, nn=False):
    df = df.dropna()
    y = get_classifier_attempt(df, nn)
    df = df.drop(columns=['posteam', 'fourth_down_converted', 'fourth_down_failed',
                 'field_goal_attempt', 'punt_attempt', 'game_date', 'down'])
    return df, y


def ready_data_go_for_it(df, nn=False):
    df = df.dropna()
    df = df[(df['fourth_down_converted'] == 1.0)
            | (df['fourth_down_failed'] == 1.0)]
    y = get_classifier_convert(df, nn)
    df = df.drop(columns=['posteam', 'fourth_down_converted', 'fourth_down_failed',
                 'field_goal_attempt', 'punt_attempt', 'game_date', 'down'])
    return df, y


def attempt_data_split(nn=False):
    return data_split(ready_data_attempt, nn)


def convert_data_split(nn=False):
    return data_split(ready_data_convert, nn)


def go_for_it_data_split(nn=False):
    return data_split(ready_data_go_for_it, nn)


def data_split(func, nn=False):
    x, y = func(pd.concat([df_2010, df_2011, df_2012, df_2013,
                df_2014, df_2015, df_2016, df_2017, df_2018]), nn)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=10)

    scaler = StandardScaler()
    scaler.fit(X_train)
    dump(scaler, '../models/scaler.joblib')
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
