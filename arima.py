# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:36:07 2023

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
"""Create Training and Test data set
fraction    choose the fraction that will be used for training, default = 0.7 (70%)
"""

def training_validation(timeseries_data, target_column, fraction = .7):
    
    train = timeseries_data[target_column][:round(len(timeseries_data[target_column]) * fraction)] # select training data
    valid = timeseries_data[target_column][round(len(timeseries_data[target_column]) * fraction):] # select validation data
    
    return train, valid

#%%
""" Build Model ARIMA(p,d,q)
train   train from training_validation
p, d, q         parameters for the model, default d=q=p=1
alpha           confidence intervals, default 0.05, 95%
transparams     default = True, checks for stationarity
"""
def build_arima(train, valid, p=1, d=1, q=1, alpha = 0.05, transparams = True):
    
    model = ARIMA(train, order=(p,d,q))  
    fitted = model.fit(disp=-1, transparams = transparams)  
    fc, se, conf = fitted.forecast(len(valid), alpha = alpha)  # 95% conf
    print(fitted.summary())
    return model, fitted, fc, conf

#%%
""" plot model results
plot for residuals, density of residuals, fitt vs actual and forecast vs actual
"""
def plot_model(fitted, train, valid, p, d, q, fc):

    # actual vs fitted
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12, 6))
    fitted.plot_predict(dynamic = False, ax = ax2) 
    ax2.set_title('model fit')
    # Plot residual errors
    residuals = pd.DataFrame(fitted.resid)
    #fig, ax = plt.subplots(1,2)
    residuals.plot(title="Residuals", ax=ax1)
    residuals.plot(kind='kde', title='Density of residual error values', ax=ax3)
    #forecast vs actual
    fc_series = pd.Series(fc, index=valid.index)
    ax4.plot(train, label='training')
    ax4.plot(valid, label='actual')
    ax4.plot(fc_series, label='forecast')
    ax4.set_title('Forecast vs Actuals')
    ax4.legend(loc='best', fontsize=8)
    plt.tight_layout()
    
    plt.savefig('arima_model{}{}{}.jpg'.format(p,d,q)) # saving figure
    return fig



