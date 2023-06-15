# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 20:12:20 2023

@author: Alexandra


Building a ARIMA Model with example data
Note! Everything you want to customize is in this file.

sources
https://hands-on.cloud/using-the-arima-model-and-python-for-time-series-forecasting/
https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
https://www.projectpro.io/article/how-to-build-arima-model-in-python/544
https://www.justintodata.com/arima-models-in-python-time-series-prediction/
"""


import os

# setting working directory
working_directory = 'C://Users//Alexandra//Documents//Studium//Master//Semester_3//region'
os.chdir(working_directory)

# Get the current working directory
current_dir = os.getcwd()

# Specify the filename
#filename = 'example.txt'
filename = "NH_Sea_Ice_Extent.txt" 

# Generate the file path by merging the current directory and filename
file_path = os.path.join(current_dir, filename)

# Print the file path
print("File path:", file_path)

#%%
""" time series needs date column and a column with the desired value
NaN values are not a problem and delt with later, 
make sure that there are not entire rows missing - not NaN but actually missing, only NaNs will be interpolated
"""
# name column of value and time/date
target_column = 'Sea_Ice_Extent' 
time_column = 'time'

#%%

import my_functions

""" function for loading your dataset as CSV 
specify:
    header = None       if first row aren't headers'; 
           = 0          if first row are headers 
    deliminter;         how the columns are separated
    skipinitialspace;   when there are additional blanks, default = True  
    timetype            if the time is given in decimal years change to 'decimal', otherwise ignore     
"""

timeseries_data = my_functions.open_timeseries_dataset(file_path = file_path, 
                                                              delimiter = ' ', 
                                                              time_column = time_column,
                                                              target_column = target_column,
                                                              timetype = 'decimal')

#%% 
"""
time-based interpolation method, which will use the timeindex to linearly interpolate NaNs
limit_direction     = 'both' will interpolate also top and bottom NaNs, other choices 'forward' (ignore top NaNs), 'backward' (ignore bottom NaNs)
inplace             = True will update the data if possible
"""
timeseries_data.interpolate(method='time', inplace = True, limit_direction = 'forward')

#%% drop any other NaNs

timeseries_data.dropna(inplace=True)

#%%

"""
Arima model needs stationary time series, following plots will help to identify needed parameters d,p and q for the model

 plot time series in original, 1st or 2nd order differenced and either autocorrelation or partial autocorrelation
 need to specify:
 order          is order of differencing
 cf             is acf for autocorrelation function, pacf for partial
 
 default parameters: if you don't specify, default is used   
 window         for rolling mean and std, default 12 months
 alpha          for confidence intervall for correlation plots, default is .05, 95% conf. int
 ax             for axes, default None then coose curent axes
 missing        how to deal with NaN, default is 'drop'    
 **plt_kwargs   you may pass any key arguments, note that they will be applied to all plots

example use:
plot_differencing(df.time, df.value, order = 2, cf = 'acf')"""

my_functions.plot_differencing(timeseries_data[time_column], timeseries_data[target_column], order = 2, cf = 'pacf')

#%%
""" 
the ARIMA Model - Auto Regressive Integrated Moving Average

how to choose p,d and q
d:  the model requires the time series to be stationary - no trends or seasonaltity. Differencing creates stationarity
    -> look at the original time series and the first and second order of Differencing plots 
    the time series is near-stationary, if the time series wiggles around a mean and the 
    autocorrelation plot reaches zero fairly quick
    If autocorrelations are positive for consecutive lags (~10) than more differencing is neccessary
    is lag 1 autocorrelaion too negative: it is over differenced, choose the lower order
    one can also look at rolling mean and std: if std is const but mean is not, than it is non-stationary

p:  the order of the AR term
    look at partial autocorrelation (correlation between time series and its lag after excluding the contributions from the intermediate lags) 
    -> p is equal to as many lags that cross the significance limit in the PACF plot
    If the PACF plot has a significant spike at lag p, but not beyond; the ACF plot decays more gradually. This may suggest an ARIMA(p, d, 0) model

q:  the order of the MA term
    look at autocorrelation plot (correlation between the time series and its lagged values),
    same as for p: q is eaqual to as many lags that cross the significance limit in the ACF plot
    If the ACF plot has a significant spike at lag q, but not beyond; the PACF plot decays more gradually. This may suggest an ARIMA(0, d, q) model

"""
#%%
#choose 
p = 4; d = 1; q = 2

import arima

#%%
"""Create Training and Test data set
fraction    choose the fraction that will be used for training, default = 0.7 (70%)
"""

train, valid = arima.training_validation(timeseries_data, target_column)

#%%

""" Build Model ARIMA(p,d,q)
training_data   train from training_validation
p, d, q         parameters for the model, default d=q=p=1
alpha           confidence intervals, default 0.05, 95%
transparams     default = True, checks for stationarity
"""


model, fitted, forecast, conf = arima.build_arima(train, valid, p = p, d = d, q = q)

""" plot model results
plot for residuals, density of residuals, fitt vs actual and forecast vs actual
"""
arima.plot_model(fitted, train, valid, p, d, q, forecast)


