# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:10:21 2023

@author: Alexandra
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime 
from datetime import timedelta
plt.rcParams['figure.dpi'] = 250 #setting global dpi for all figures

#%%
""" specify:
    header              = None if first row aren't headers'; 
                        = 0    if first row are headers 
    deliminter;         how the columns are separated
    skipinitialspace;   when there are additional blanks, default = True    
    format              for datetime format, default year-month-day
    timetype            if the time is given in decimal years change to 'decimal', otherwise ignore
"""
def open_timeseries_dataset(file_path, delimiter, time_column, target_column, dformat = '%Y-%m-%d', header = 0, skipinitialspace = True, timetype = None):
       
    try:
        dataset = pd.read_csv(file_path, delimiter = delimiter, header = header, skipinitialspace = skipinitialspace)  
        dataset.columns = [time_column, target_column]
    except pd.errors.EmptyDataError:
        print("Error: The dataset is empty.") #check if dataset is empty
        return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.") # check if the path is working
        return None
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file. Please check the file format.")


# try to deal with different time columns
    if timetype == 'decimal': #years in decimal

        for i in range(len(dataset[time_column])):
            dataset[time_column][i] = datetime(int(dataset[time_column][i]), 1, 1) + timedelta(days = (dataset[time_column][i] % 1) * 365)
            dataset.set_index(dataset[time_column], inplace = True) # setting datetime as index, inplace = True updates dataset
    # elif 'year' in dataset.columns and 'month' in dataset.columns: # one column with year and one with month 
    #     # converting year and month column to one datetime format
    #     dataset[time_column] = pd.to_datetime(dataset['year'].astype(str) + dataset['month'].astype(str), format='%Y%m').dt.date # change dt.date if you want to keep time    

    #just try to read in the time as datetime
    else:
        try:    
            dataset[time_column] = pd.to_datetime(dataset[time_column], dformat) # convert to datetime using desired format     
            dataset.set_index(dataset[time_column], inplace = True) # setting datetime as index, inplace = True updates dataset
        except:
            print("Some Error with to_datetime ... try strptime instead") 
        
            try:
                dataset[time_column] = datetime.strptime(dataset[time_column], dformat).date() #convert to datetime object using strptime
                dataset.set_index(dataset[time_column], inplace = True) # setting datetime as index, inplace = True updates dataset
            except:
                print("Some other datetime error") 
     
    return dataset

#%%
    """ plot time series in original, 1st or 2nd order differenced and either autocorrelation or partial autocorrelation
 need to specify:
 order          is order of differencing
 cf             is acf for autocorrelation function, pacf for partial
 
 default parameters: if you don't specify, default is used   
 window         for rolling mean and std, default 12 months
 alpha          for confidence intervall for correlation plots, default is .05, 95% conf. int
 ax             for axes, default None then coose curent axes
 missing        how to deal with NaN, default is 'drop'   
periods         period for differencing, default 12 months for seasonal differencing
 **plt_kwargs   you may pass any key arguments, note that they will be applied to all plots"""
 
 
"""to be called in plot_differencing 
 create function that plots time series original, 1st or 2nd order differenced with rolling mean and rolling std 
 arguments same as in plot_differencing"""
 
def plot_series(x, y, order, window, periods, ax = None, **plt_kwargs ):
    if ax is None:
        ax = plt.gca()
        
    if order == 0:

        label = 'Original'
        ax.plot(x, y.dropna(), label = label, **plt_kwargs) # plot timeseries of variable 
        ax.set_title(label)
        ax.plot(x, y.dropna().rolling(window).mean(), label = 'rolling mean', **plt_kwargs)
        ax.plot(x, y.dropna().rolling(window).std(), label = 'rolling std', **plt_kwargs)
        ax.legend()
        plt.tight_layout()
        
    if order == 1:
        label = '1st order Differencing'
        ax.plot(x, y.dropna().diff(periods), label = label, **plt_kwargs) # plot timeseries of variable 
        ax.set_title(label)
        ax.plot(x, y.dropna().diff(periods).rolling(window).mean(), label = 'rolling mean', **plt_kwargs)
        ax.plot(x, y.dropna().diff(periods).rolling(window).std(), label = 'rolling std', **plt_kwargs)
        ax.legend()
        plt.tight_layout()
        
    if order == 2:
        label = '2nd order Differencing'
        ax.plot(x, y.dropna().diff(periods).diff(periods), label = label, **plt_kwargs) # plot timeseries of variable 
        ax.set_title(label)
        ax.plot(x, y.dropna().diff(periods).diff(periods).rolling(window).mean(), label = 'rolling mean', **plt_kwargs)
        ax.plot(x, y.dropna().diff(periods).diff(periods).rolling(window).std(), label = 'rolling std', **plt_kwargs)
        ax.legend()
        plt.tight_layout()    
    return ax

"""to be called in plot_differencing 
plots either autocorrelation or partial autocorrelation"""
    
def plot_cf(y, cf, ax, alpha, missing):
    conf = 1 - alpha
    if cf == 'acf':         #autocorrelation
        plot_acf(y, ax, alpha = alpha, missing = missing, title = 'Autocorrelation with conf. Interval {}%'.format(conf))
    else:
        plot_pacf(y, ax, alpha = alpha, title = 'Partialautocorrelation with conf. Interval {}%'.format(conf))    # partial autocorrelation
    return ax
               

"""
example use:
plot_differencing(df.time, df.value, order = 2, cf = 'acf')"""

        
def plot_differencing(x, y, order, cf, alpha = .05, missing = 'drop', periods = 12, window = 12, ax = None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
        
    if order == 0:

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 6))    # creates subplots
        plot_series(x, y, order, window, periods, ax = ax1)              # plots time series
        plot_cf(y.dropna(), cf = cf, alpha = alpha, missing = missing, ax = ax2)                                # plots correlation
            
    if order == 1:
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7), sharey = False) # same y axis for subplots in one column 'col
        plot_series(x, y, 0, window, periods, ax = ax1)
        plot_cf(y, cf = cf, alpha = alpha, missing = missing, ax = ax2)
        
        plot_series(x, y, order, window, periods, ax = ax3)
        plot_cf(y.diff(periods).dropna(), cf = cf, alpha = alpha, missing = missing, ax = ax4) 

        
    if order == 2:
            
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 8), sharey = False) 
        plot_series(x, y, 0, window, periods, ax = ax1)
        plot_cf(y, cf = cf, alpha = alpha, missing = missing, ax = ax2)
       
        plot_series(x, y, 1, window, periods, ax = ax3)
        plot_cf(y.diff(periods).dropna(), cf = cf, alpha = alpha, missing = missing, ax = ax4) 
        
        plot_series(x, y, order, window, periods, ax = ax5)
        plot_cf(y.diff(periods).diff(periods).dropna(), cf = cf, alpha = alpha, missing = missing, ax = ax6)
        
    plt.savefig('Differencing_{}.jpg'.format(cf)) # saving figure
    return ax


