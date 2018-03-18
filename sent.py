import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from textblob import TextBlob


dataset = pd.read_csv('tweets_gathered1.csv',encoding = "ISO-8859-1")
dataset1 = pd.read_csv('TCS.csv')
corpus = []
date=[]
date1=[]
for i in range(0, 14354):
    corpus.append(dataset['text'][i])
    time = dataset['date'][i]
    time = time.split(' ')
    date.append(time[0])

for i in range(0,300):
    date1.append(dataset1['#NAME?'][i])
    
myDictPol = {}
myDictCount ={}

#initialization
for i in range(300):
    
        myDictPol[date1[i]] = 0
        myDictCount[date1[i]] = 0

#compute        
for i in range(7280):
    
    text=corpus[i]
    blob = TextBlob(text)

    for sentence in blob.sentences:
        pol=(sentence.sentiment.polarity)
    
    if date[i] in myDictPol.keys():         
        myDictPol[date[i]] += pol
        myDictCount[date[i]] += 1

for key in myDictPol.keys():
    if myDictCount[key]!=0:
        myDictPol[key] /= myDictCount[key]

# Importing the dataset
datasetf = pd.read_csv('mercury_TCS.csv')
X = datasetf.iloc[300, [0,1,2,3,4]].values
Y = datasetf.iloc[300,5].values

arr=[]
for key in myDictPol.keys():
    arr.append(myDictPol[key])

Diff_Y = np.zeros(len(Y))
Senti_X = arr[:]

for i in range(1,len(Y)):
    Diff_Y[i] = Y[i] - Y[i-1]


from sklearn.cross_validation import train_test_split
Senti_X_train,Senti_X_test,Diff_Y_train,Diff_Y_test = train_test_split(Senti_X,Diff_Y,test_size=1/3,random_state=0)

'''
Senti_X_train = Senti_X_train.reshape(len(Senti_X_train),1)
Senti_X_test = Senti_X_test.reshape(len(Senti_X_test),1)

Diff_Y_train = Diff_Y_train.reshape(len(Diff_Y_train),1)
Diff_Y_test = Diff_Y_test.reshape(len(Diff_Y_test),1)
'''

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(Senti_X_train, Diff_Y_train)

Diff_Y_pred = regressor.predict(Senti_X_test)

regressor.score(Senti_X_test,Diff_Y_test)

plt.scatter(Senti_X_train,Diff_Y_train,color='red')
plt.show()
