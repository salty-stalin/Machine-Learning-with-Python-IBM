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

print(test)


#########################
#########################
#Fitting Model
#########################
#########################




""" skikit uses ordinary least squares method (OLS) to fit the data

OLS chooses the parameters of a linear function of a set of explanatory variables by minimizing the sum of 
the squares of the differences between the target dependent variable and those predicted by the linear function.
 In other words, it tries to minimizes the sum of squared errors (SSE) or mean squared error (MSE)
 between the target variable (y) and our predicted output ( 𝑦̂  ) over all samples in the dataset.

OLS can find the best parameters using of the following methods: - Solving the model parameters 
analytically using closed-form equations - Using an optimization algorithm
(Gradient Descent, Stochastic Gradient Descent, Newton’s Method, etc.

"""

from sklearn import linear_model
regr = linear_model.LinearRegression() #select linear regression model
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]) #select x values
y = np.asanyarray(train[['CO2EMISSIONS']]) #select y-values
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)



#########################
#########################
#PREDICTION
#########################
#########################

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]) #Predict the y-values for the following x-values
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]) #select x-test data
y = np.asanyarray(test[['CO2EMISSIONS']]) #slect y-test data
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


"""explained variance score 

If  𝑦̂   is the estimated target output, y the corresponding (correct) target output, and Var is Variance,
 the square of the standard deviation, then the explained variance is estimated as follow:

𝚎𝚡𝚙𝚕𝚊𝚒𝚗𝚎𝚍𝚅𝚊𝚛𝚒𝚊𝚗𝚌𝚎(𝑦,𝑦̂ )= 1−Var{𝑦−𝑦̂ }/𝑉𝑎𝑟{𝑦}
 
The best possible score is 1.0, lower values are worse.
"""


