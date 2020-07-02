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



#########################
#########################
#FITTING POLYNOMIAL REGRESSION
#########################
#########################

""" to fit a polynomial such us of order 2 we use PolynomialFeatures()

PloynomialFeatures() function in Scikit-learn library, drives a new feature sets from the original feature set.
 That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less
 than or equal to the specified degree. For example, lets say the original feature set has only one feature, ENGINESIZE. 
 Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2
 
 e.g   
 [x1]   goes to [1,x1,x1^2]          [2]    goes to [1,2,4]
 [x2]           [1,x2,x2^2]          [3]            [1.3.9]
 [x3]           [1,x3,x3^2]          [4]            [1,4,16]
 """
 
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']]) #select train data
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']]) #select test data
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x) #create polynomial matrix array of x values

"""we  write the polynomial as a simple linear regression with the new values 𝑦=𝑏+𝜃1𝑥1+𝜃2𝑥2
and treat this as a linear regression problem. So we can use LinearRegression function to solve it"""

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#########################
#########################
#EVALUATION
#########################
#########################

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )




########################
########################
#TRY 3RD DEGREE POLYNOMIAL FOR FITTING
#######################
#######################

poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)
# The coefficients
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ',clf3.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf3.intercept_[0]+ clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX, 2) + clf3.coef_[0][3]*np.power(XX, 3)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
test_x_poly3 = poly3.fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )
