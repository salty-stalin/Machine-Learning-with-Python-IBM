import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import wget
url= 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
#filename=wget.download(url)

df = pd.read_csv("FuelConsumptionCo2.csv") #read datafile

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] #read selected columns

#########################
#########################
#Creating train and test dataset
#########################
#########################

"""Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually
exclusive. After which, you train with the training set and test with the testing set. This will provide a more
accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have
been used to train the data. It is more realistic for real world problems.


Lets split our dataset into train and test sets, 80% of the entire data for training, and the 20% for testing. We
create a mask to select random rows using np.random.rand() function:

"""
#Split data set 80% training 20% testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

""" Fitting simple regression model 
Linear Regression fits a linear model with coefficients  𝜃=(𝜃1,...,𝜃𝑛)  to minimize the 'residual sum of squares' between the
independent x in the dataset, and the dependent y by the linear approximation."""

#Train Data Distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Modelling using sklean package

from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])  #select x training dataset
train_y = np.asanyarray(train[['CO2EMISSIONS']]) #select y training dataset
regr.fit (train_x, train_y)  #fit the regression model
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

#Plot Outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#########################
#########################
#EVALUATION
#########################
#########################
"""There are different model evaluation metrics, lets use MSE here to calculate the accuracy of our model based on the test set:

-Mean absolute error: It is the mean of the absolute value of the errors. 
This is the easiest of the metrics to understand since it’s just average error.

-Mean Squared Error (MSE): Mean Squared Error (MSE) is the mean of the squared error. It’s more popular than Mean
absolute error because the focus is geared more towards large errors. This is due to the squared term exponentially
increasing larger errors in comparison to smaller ones.

-Root Mean Squared Error (RMSE): This is the square root of the Mean Square Error.

-R-squared is not error, but is a popular metric for accuracy of your model. It represents how close the data are to the
fitted regression line. The higher the R-squared, the better the model fits your data. Best possible score is 1.0 and it can be
negative (because the model can be arbitrarily worse). """

from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']]) #select x testing dataset
test_y = np.asanyarray(test[['CO2EMISSIONS']]) #select y testing dataset
test_y_hat = regr.predict(test_x) #predicted y for given x


print(test_x)
print(test_y)
print(test_y_hat)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
