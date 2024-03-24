import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn import metrics
from statsmodels.tsa.ar_model import AutoReg


def read_rew_data(folder_path):
    # load the dataframe of energy and weather frome the csv files
    electricity_data_2017 = pd.read_csv(folder_path + 'IST_Civil_Pav_2017.csv', index_col='Date_start')
    electricity_data_2017.index = pd.to_datetime(electricity_data_2017.index, format='%d-%m-%Y %H:%M')

    electricity_data_2018 = pd.read_csv(folder_path + 'IST_Civil_Pav_2018.csv', index_col='Date_start')
    electricity_data_2018.index = pd.to_datetime(electricity_data_2018.index, format='%d-%m-%Y %H:%M')

    electricity_df = pd.concat([electricity_data_2017, electricity_data_2018])

    weather_df = pd.read_csv(folder_path + 'IST_meteo_data_2017_2018_2019.csv',
                             parse_dates=['yyyy-mm-dd hh:mm:ss'],
                             # Replace 'Date_Column_Name' with your actual date column name
                             index_col='yyyy-mm-dd hh:mm:ss')

    # Aggregate weather data to hourly by taking the mean
    df_weather_hourly = weather_df.resample('h').mean()

    # joining the energy and weather data in the same hourly dataframe
    df_data = electricity_df.join(df_weather_hourly, how='outer')

    df_data['hour'] = df_data.index.hour

    df_data['Season'] = df_data.index.month.map(get_season)

    df_training = df_data[~(df_data.index.year == 2019)]
    df_training = get_power_1(df_training)
    df_training = get_difference(df_training)

    return df_training


# Function to determine the season based on the month
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    elif month in [9, 10, 11]:
        return 3  # Fall


# Function to determine if a day is a weekday or weekend
def day_type(date):
    if date.weekday() < 5:
        return 1
    else:
        return 0


def get_power_1(df_training):
    # Calculate the time differences between consecutive rows
    time_diffs = df_training.index.to_series().diff()

    # Initialize a new column with NaN values
    df_training['Power-1'] = np.nan
    nb_col = len(df_training.columns)
    # Loop through the DataFrame to fill 'shifted_column'
    for i in range(1, len(df_training)):
        # If the time difference is exactly 1 hour, shift the value
        if time_diffs.iloc[i] == pd.Timedelta(hours=1):
            df_training.iloc[i, nb_col - 1] = df_training.iloc[i - 1, 0]

    return df_training


def get_difference(df_training):
    time_diffs = df_training.index.to_series().diff()

    # Initialize a new column for storing differences
    df_training['Power-diff-1'] = np.nan
    nb_col = len(df_training.columns)
    # Loop through the DataFrame starting from the second row (Python index 2)
    for i in range(2, len(df_training)):
        # Check if both time differences, (i) to (i-1) and (i-1) to (i-2), are less than or equal to 1 hour
        if time_diffs.iloc[i] <= pd.Timedelta(hours=1) and time_diffs.iloc[i - 1] <= pd.Timedelta(hours=1):
            # Calculate the difference in 'Power_kW' values between (i-1) and (i-2) and assign it
            df_training.iloc[i, nb_col - 1] = df_training.iloc[i - 1, 0] - df_training.iloc[i - 2, 0]

    del time_diffs, i
    # If any of the time differences is more than 1 hour, 'Power-diff-1' remains NaN for that row

    df_training = df_training[~(df_training['Power-diff-1'].isna())]

    return df_training


def get_result_forecast(folder_path):
    result_forecast2019 = pd.read_csv(folder_path + 'result_forecast2019.csv', index_col='Date')
    result_forecast2019.index = pd.to_datetime(result_forecast2019.index, format='%Y-%m-%d %H:%M:%S')
    return result_forecast2019


def autoreg_benchmark(train, test):
    window = 1
    train = train[~train.isna()]
    test = test[~test.isna()]
    model = AutoReg(train, lags=1)
    model_fit = model.fit()

    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train) - window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef[0]
        for d in range(window):
            yhat += coef[d + 1] * lag[window - d - 1]
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    return predictions


def error_df(pred_df,keys=['test', 'prediction', 'prediction_AR']):
    test, predictions, prediction_AR = pred_df[keys[0]].sort_index(), pred_df[keys[1]].sort_index(), pred_df[keys[2]].sort_index()

    MAE_AR = metrics.mean_absolute_error(test, prediction_AR)
    MBE_AR = np.mean(test - prediction_AR)
    MSE_AR = metrics.mean_squared_error(test, prediction_AR)
    RMSE_AR = np.sqrt(metrics.mean_squared_error(test, prediction_AR))
    cvRMSE_AR = RMSE_AR / np.mean(test)
    NMBE_AR = MBE_AR / np.mean(test)

    MAE = metrics.mean_absolute_error(test, predictions)
    MBE = np.mean(test - predictions)
    MSE = metrics.mean_squared_error(test, predictions)
    RMSE = np.sqrt(metrics.mean_squared_error(test, predictions))
    cvRMSE = RMSE / np.mean(test)
    NMBE = MBE / np.mean(test)

    error = pd.DataFrame({'Model':['Best model', 'AutoRegression'],
                        'MAE': [MAE, MAE_AR],
                        'MBE': [MBE, MBE_AR],
                        'MSE': [MSE, MSE_AR],
                        'RMSE': [RMSE, RMSE_AR],
                        'cvRMSE': [cvRMSE, cvRMSE_AR],
                        'NMBE': [NMBE, NMBE_AR]})
    return error