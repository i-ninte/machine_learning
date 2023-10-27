import numpy as np
import pandas as pd
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load your time series data into a pandas Series
# Replace 'your_time_series_data.csv' and 'column_name' with your actual data
data = pd.read_csv('your_time_series_data.csv')['column_name']

# Stationarity check
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=52, center=False).mean()
    rolstd = timeseries.rolling(window=52, center=False).std()

    orig = plt.plot(timeseries, color='blue', label="Original")
    mean = plt.plot(rolmean, color='red', label="Rolling Mean")
    std = plt.plot(rolstd, color='black', label="Rolling Std")
    plt.legend(loc="best")
    plt.show(block=False)

    # Dickey Fuller Test
    print("Dickey Fuller test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(dftest[0:4], index=['Test statistic', 'p-value', '#lags Used', 'Number of observations used'])
    for key, value in dftest[4].items():
        dfoutput[f'critical value {key}'] = value

    print(dfoutput)

# Stationarity check
test_stationarity(data)

# Decompose by log decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data, model='multiplicative')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot decomposed components
plt.subplot(411)
plt.plot(data, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')
plt.tight_layout()

# Find p and q parameters using ACF and PACF
lag_acf = acf(residual, nlags=20)
lag_pacf = pacf(residual, nlags=20, method='ols')

# Plot ACF
plt.subplot(121)
plt.stem(lag_acf)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=-1.96 / np.sqrt(len(residual)), color='gray', linestyle='--')
plt.axhline(y=1.96 / np.sqrt(len(residual)), color='gray', linestyle='--')
plt.title('ACF')

# Plot PACF
plt.subplot(122)
plt.stem(lag_pacf)
plt.axhline(y=0, color='gray', linestyle='--')
plt.axhline(y=-1.96 / np.sqrt(len(residual)), color='gray', linestyle='--')
plt.axhline(y=1.96 / np.sqrt(len(residual)), color='gray', linestyle='--')
plt.title('PACF')

plt.tight_layout()

# Determine p and q values based on ACF and PACF plots
# For example, ARIMA(2, 1, 1) would have p=2, d=1, and q=1
p = 2
d = 1
q = 1
model = sm.tsa.ARIMA(data, order=(p, d, q))
results = model.fit()

# Make predictions with the ARIMA model
# For example, you can use results.forecast(steps=n) to forecast n time steps into the future
