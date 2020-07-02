import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import matplotlib.pyplot as plt
import wget


url= 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/teleCust1000t.csv'
#filename=wget.download(url)

df=pd.read_csv('teleCust1000t.csv')
print(df.head())
print(df['custcat'].value_counts()) #see number of counts of values of custcat column (our classification)
df.hist(column='income', bins=50)
plt.show()


#need to define feature set X

df.columns

#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
print(X[0:5])

#define labels

y = df['custcat'].values
y[0:5]

#################
#################
##NORMALISE DATA
#################
#################

"""
Data Standardization give data zero mean and unit variance, it is good practice,
 especially for algorithms such as KNN which is based on distance of cases:
"""
     
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])


#################
#################
##APPLY TRAIN TEST SPLIT
#################
#################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



#################
#################
##KNN TRAINING
#################
#################

#Import library

from sklearn.neighbors import KNeighborsClassifier

#train with algorith k=4

k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


#################
#################
##KNN PREDICTING
#################
#################

yhat = neigh.predict(X_test)
yhat[0:5]

"""Need to test accuracy evaluation
This is equvelant to jaccard simmilarity score, essentially it calculates how closely 
the actual labels are matched in the data set

"""

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))