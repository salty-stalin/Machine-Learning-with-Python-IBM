import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wget

url= 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv'
filename=wget.download(url)

df=pd.read_csv("china_gdp.csv")
print(df.head(10))

#Plotting the Dataset

plt.figure(figsize=(8,5))
x_data, y_data = (df["Year"].values, df["Value"].values)
plt.plot(x_data,y_data,'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#Choosing a model

""" It seems that the sigmoid/logistic model fits the data well
    
    this has the for of Y= 1/[1+e^Beta_1(x-Beta_2)]          x= independent variable
    
    Beta_1- controls the curves steepness
    Beta_2- slides the the curve on the x-axis 
"""
    
#Define function for sigmoid

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
 
""" The task to find the best parameters to fit the data. This can be done using 
    curve_fit, which uses non-linear least squares to fit the sigmoid function
    
    Optimal values for the parameters so that the sum of the squared residuals 
    of sigmoid(xdata, *popt) - ydata is minimized.
"""

#First need to normalise the data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

#Applying curve_fit
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#Plotting the regression Model

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

##################
##################
#EVALUATION
##################
##################

from sklearn.metrics import r2_score
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , test_y) )

