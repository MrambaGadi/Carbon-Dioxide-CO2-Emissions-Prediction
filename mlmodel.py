import numpy as np
'''
This is a simple linear regression model to predit the CO2 emmission from cars
Dataset:
FuelConsumption.csv, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions
for new light-duty vehicles for retail sale in Canada
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
#df.head()
df["FUELTYPE"] = np.where(df["FUELTYPE"]=="Z", "1", df["FUELTYPE"])
df["FUELTYPE"] = np.where(df["FUELTYPE"]=="X", "2", df["FUELTYPE"])
df["FUELTYPE"] = np.where(df["FUELTYPE"]=="D", "3", df["FUELTYPE"])
df["FUELTYPE"] = np.where(df["FUELTYPE"]=="E", "4", df["FUELTYPE"])
#use required features
cdf = df[['ENGINESIZE','CYLINDERS','FUELTYPE','CO2EMISSIONS']]

#Training Data and Predictor Variable
# Use all data for training (tarin-test-split not used)
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]


regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(regressor, open('model.pkl','wb'))

'''
#Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2.6, 8, 10.1]]))
'''
