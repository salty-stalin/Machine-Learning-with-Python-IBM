import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import wget

url= 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv'
filename=wget.download(url)

churn_df = pd.read_csv("ChurnData.csv")
print(churn_df.head())

#Select the Columns
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']] 
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

#Define X and Y for the dataset

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

y = np.asarray(churn_df['churn'])
y [0:5]

#Normalise Dataset

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#Split Dataset into Train and test Set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#################
#################
##MODELLING
#################
#################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#C parameter indicates inverse of regularization strength which must be a positive float. Smaller values specify stronger regularization
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

#Predict test set
yhat = LR.predict(X_test)
print(yhat)
"""
predict_proba returns estimates for all classes, ordered by the label of classes. 
So, the first column is the probability of class 1, P(Y=1|X), and second column 
is probability of class 0, P(Y=0|X): """

yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

#################
#################
##EVALUATION
#################
#################

""" Jaccard Index for Evaluation 

. we can define jaccard as the size of the intersection divided by the size of the union
 of two label sets. If the entire set of predicted labels for a sample strictly match with 
 the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.

"""

from sklearn.metrics import jaccard_similarity_score
jaccard_evaluation=jaccard_similarity_score(y_test, yhat)
print(jaccard_evaluation)

""" Log loss Evaluation

This probability is a value between 0 and 1. Log loss( Logarithmic loss) measures
 the performance of a classifier where the predicted output is a probability value
 between 0 and 1."""
 
from sklearn.metrics import log_loss
log_loss_evaluation=log_loss(y_test, yhat_prob)
print(log_loss_evaluation)

"""Confusion Matrix Evaluation
Look at first row. The first row is for customers whose actual churn value in test set is 1.
 As you can calculate, out of 40 customers, the churn value of 15 of them is 1. And out of these 15,
 the classifier correctly predicted 6 of them as 1, and 9 of them as 0.

It means, for 6 customers, the actual churn value were 1 in test set, and classifier 
also correctly predicted those as 1. However, while the actual label of 9 customers were 1,
 the classifier predicted those as 0, which is not very good. We can consider it 
 as error of the model for first row.

What about the customers with churn value 0? Lets look at the second row. 
It looks like there were 25 customers whom their churn value were 0.

The classifier correctly predicted 24 of them as 0, and one of them wrongly as 1. 
So, it has done a good job in predicting the customers with churn value 0. 
A good thing about confusion matrix is that shows the model’s ability to correctly
 predict or separate the classes. In specific case of binary classifier, 
 such as this example, we can interpret these numbers as the count of true positives,
 false positives, true negatives, and false negatives.
"""

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
print (classification_report(y_test, yhat))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')