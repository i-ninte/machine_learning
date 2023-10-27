#building the model 
#steps 
'''
find trends 
make it stationery
decompose by log decomposition
find p, q parameters 
thus Auto regressive values and moving average 
then build an arima model thus auto regressive integrated moving average model.
make predictions with the arima model.

'''
import numpy as np
import pandas as pd
fromm datetime import datetime as dt
from statsmodel.tsa.statstools import adfuller, acf, pacf
from statsmodel.tsa.statstools.arima_model import ARIMA
import math
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6


#stationaity check
def test_stationaity(timeseries):
	rolmean= timeseries.rolling(window=52, center=False).mean()
	rolstd= timeseries.rolling(window=52, center=False).std
	
	orig= plt.plot(timeseries, color='blue',label="Original")
	mean= plt.plot(rolmean, color='red', label="RollingMean")
	std= plt.plot(rolstd, color='black', label="RollingStd")
	plt.legend(loc"best")
	plt.show(block=False)
	
	
	
	
	
	#Dickey Fuller Test
	print("dickey fuller test: ")
	dftest= adfuller(timeseries, autolag="AIC")
	dfoutput= pd.Series(dftest[0:4], index=['Test statistic', 'p-value', '#lags Used', 'Number of observations used'])
	for key,value in dftest[4].items():
		dfoutput['critical value %key]= value
					
	print(dfoutput)
					
