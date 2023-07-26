#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the huber regression model on the training set
regressor = HuberRegressor(epsilon=1.35)
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print('Performance metrics on the testing set:')
print('Actual values:',y_test)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)
print('Predicted values:',y_pred)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Huber Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[6]:


import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the robust regression model on the training set
regressor = RANSACRegressor(min_samples=50)
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Performance metrics on the testing set:')
print('Actual values:',y_test)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)
print('Predicted values:',y_pred)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Robust Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[8]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the random forest regression model on the training set
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print('Performance metrics on the testing set:')
print('Actual values:',y_test)
print('Predicted values:', y_pred)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Random Forest Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[9]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the gradient boosting regression model on the training set
regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print('Performance metrics on the testing set:')
print('Actual values:', y_test)
print('Predicted values:', y_pred)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Gradient Boosting Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Build the neural network model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae', 'mse'])

# Train the model on the training set
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, y_test), verbose=0)

# Predict the target variable for the testing set using the trained model
y_pred = model.predict(X_test).flatten()

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print('Performance metrics on the testing set:')
print('Actual values:', y_test)
print('Predicted values:', y_pred)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Neural Network Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[17]:


import pandas as pd
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the target variable for training set
y_train = train_df.iloc[:, -1].values

# Train the ARIMA model on the training set
model = ARIMA(y_train, order=(3, 1, 1))
model_fit = model.fit()

# Predict the target variable for the testing set using the trained model
y_pred = model_fit.forecast(steps=12)[0]

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(test_df.iloc[:, -1], y_pred)
mad = mean_absolute_error(test_df.iloc[:, -1], y_pred)
r2 = r2_score(test_df.iloc[:, -1], y_pred)

print('Performance metrics on the testing set:')
print('Actual values:', test_df.iloc[:, -1].values)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)
print('Predicted values:', y_pred)

# Plot the actual and predicted values
plt.plot(test_df.index, test_df.iloc[:, -1], label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'ARIMA: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[22]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Split the dataset into train and test sets
train_data = df.iloc[:-12]
test_data = df.iloc[-12:]

# Fit the ARIMA model to the training data
model = ARIMA(train_data['cpu_utilization'], order=(1,0,0))
model_fit = model.fit()

# Predict the CPU utilization for the test data using the fitted model
predictions = model_fit.forecast(steps=12)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(test_data['cpu_utilization'], predictions)
mad = mean_absolute_error(test_data['cpu_utilization'], predictions)
r2 = r2_score(test_data['cpu_utilization'], predictions)

print('Performance metrics on the testing set:')
print('Actual values:',test_data['cpu_utilization'].values)
print('Predicted values:',predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)

# Plot the actual and predicted values
plt.plot(test_data.index, test_data['cpu_utilization'], label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'ARIMA Model: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[21]:


import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data from a CSV file
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Split the data into training and testing sets
train_df = df[:len(df)-12]
test_df = df[len(df)-12:]

# Create an ARIMA model with order (1,1,1)
model = ARIMA(train_df, order=(1,1,1))

# Fit the model to the training data
model_fit = model.fit()

# Predict the CPU utilization for the test data using the fitted model
predictions = model_fit.forecast(steps=12)[0]

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(test_df, predictions)
mad = mean_absolute_error(test_df, predictions)
r2 = r2_score(test_df, predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)


# In[23]:


import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Split the dataset into train and test sets
train_data = df.iloc[:-12]
test_data = df.iloc[-12:]

# Extract the univariate time series data from the train_data DataFrame
endog = train_data['cpu_utilization']

# Fit the SARIMAX model to the training data
model = SARIMAX(endog=endog, order=(1,0,0), seasonal_order=(1,1,1,12))
model_fit = model.fit()

# Predict the CPU utilization for the test data using the fitted model
predictions = model_fit.forecast(steps=12)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(test_data['cpu_utilization'], predictions)
mad = mean_absolute_error(test_data['cpu_utilization'], predictions)
r2 = r2_score(test_data['cpu_utilization'], predictions)

print('Performance metrics on the testing set:')
print('Actual values:',test_data['cpu_utilization'].values)
print('Predicted values:',predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)

# Plot the actual and predicted values
plt.plot(test_data.index, test_data['cpu_utilization'], label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'SARIMAX Model: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[24]:


import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-58:, :]

# Get the remaining rows for training
train_df = df.iloc[:-58, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the huber regression model on the training set
regressor = HuberRegressor(epsilon=1.35)
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print('Performance metrics on the testing set:')
print('Actual values:',y_test)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)
print('Predicted values:',y_pred)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Huber Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[25]:


import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-58:, :]

# Get the remaining rows for training
train_df = df.iloc[:-58, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the robust regression model on the training set
regressor = RANSACRegressor(min_samples=50)
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Performance metrics on the testing set:')
print('Actual values:',y_test)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)
print('Predicted values:',y_pred)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Robust Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[31]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data.csv', skiprows=1)

# Get the last 12 rows for testing
test_df = df.iloc[-12:, :]

# Get the remaining rows for training
train_df = df.iloc[:-12, :]

# Extract the input features and the target variable for training set
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

# Extract the input features and the target variable for testing set
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

# Train the linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the target variable for the testing set using the trained model
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model on the testing set using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(y_test, y_pred)
mad = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print('Performance metrics on the testing set:')
print('Actual values:',y_test)
print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)
print('Predicted values:', y_pred)

# Plot the actual and predicted values
plt.plot(test_df.index, y_test, label='Actual')
plt.plot(test_df.index, y_pred, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'Linear Regression: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[30]:


import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset into a pandas DataFrame
df = pd.read_csv('cpu_data3.csv', skiprows=1)

# Split the dataset into train and test sets
train_data = df.iloc[:-12]
test_data = df.iloc[-12:]

# Fit the ARIMA model to the training data
model = ARIMA(train_data['cpu_utilization'], order=(1,0,0))
model_fit = model.fit()

# Predict the CPU utilization for the test data using the fitted model
predictions = model_fit.forecast(steps=12)

# Evaluate the performance of the model using mean squared error, mean absolute deviation, and R-squared score
mse = mean_squared_error(test_data['cpu_utilization'], predictions)
mad = mean_absolute_error(test_data['cpu_utilization'], predictions)
r2 = r2_score(test_data['cpu_utilization'], predictions)

print('Performance metrics on the testing set:')
print('Actual values:',test_data['cpu_utilization'].values)
print('Predicted values:',predictions)

print('Mean Squared Error:', mse)
print('Mean Absolute Deviation:', mad)
print('R-squared Score:', r2)

# Plot the actual and predicted values
plt.plot(test_data.index, test_data['cpu_utilization'], label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.xlabel('Time Interval')
plt.ylabel('CPU Utilization')
plt.title(f'ARIMA Model: MSE={mse:.2f}, MAD={mad:.2f}, R-squared={r2:.2f}')
plt.legend()
plt.show()


# In[ ]:




