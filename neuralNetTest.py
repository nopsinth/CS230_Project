import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textblob import TextBlob
import time
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from random import randrange


startTime = time.time()
#time.sleep(61)
#print('Total Time: ' + str(int((time.time()-startTime)/60)) + ' minutes ' + str((time.time()-startTime)%60.0) + " seconds")

Y = list()
X = list()
numdict = defaultdict(list)
data = defaultdict(list)
TOD = defaultdict(list)
with open('processedIBDinfo.txt',encoding='utf-8') as f: lineList = f. readlines()
f.close()
f = lineList

for line in f:
    parts = line.split('/t')
    username = parts[6]
    post = parts[3]
    data[username].append(post)
    time =  parts[5].replace(':','')
    try:
        time = time.split()[1]
    except:
        if(parts[4]=="Post or Reply"):
            continue
        try:
            time = parts[4].replace(':','').split()[1]
        except:
            time = parts[3].replace(':','').split()[1]
    TOD[username].append(time)


def addFeatureByDiv1(d):
    n = 0
    for user in d:
        if(n==0):
            n+=1
            continue
        total = 0
        num = 0
        for post in d[user]:
            total+= TextBlob(post).sentiment.polarity
            num+=1
        numdict[user].append(total/float(num))
        n+=1

def addFeatureByDiv2(d):
    #time of day they post at
    for user in d:
        if(user == 'UserID'):
            continue
        total = 0
        num = 0
        for post in d[user]:
            try:
                total+= float(post)
            except:
                continue
            num+=1
        try:
            numdict[user].append((total/float(num))/2359.0)
        except:
            numdict[user].append(0.0)

#addFeatureByDiv1(data)

#addFeatureByDiv2(TOD)

userDictX = defaultdict(list)
userDictY = defaultdict(list)
first = True
drugNames = None
for thing in open('userInfo.txt','r'):
    if(first):
        first=False
        drugNames = thing.split('/t')[10:]
        continue
    info = thing.split('/t')
    features = info[2:10]
    username = info[0]
    Yval = info[11:]
    yvt = list()
    for thing in Yval:
        if(thing.lower().strip()=='yes'):
            yvt.append(1)
        else:
            yvt.append(0)
    userDictY[username] = yvt
    xvt = list()
    age = None
    for x in range(len(features)):
        if(x==0):
            xvt.append(int(features[x]))
        if(x==1):
            xvt.append(int(features[x]))
        if(x==2):
            xvt.append(int(features[x].split('/')[0])/float(12.0)) #how far into the year
            xvt.append(int(features[x].split('/')[1])/float(30.0)) #how far into the month
            xvt.append(int(features[x].split('/')[2])-2008) #how long after the forums were created were they posted
        if(x==3):
            xvt.append(int(features[x]=='Male')) #if they are male
        if(x==4):
            age = 2019-int(features[x])
            xvt.append(2019-int(features[x])) #age
        if(x==5):
            if(not (features[x]=='Patient')): #skips if not a patient
                continue
        if(x==6):
            if("crohn" in features[x].lower()): #if they have crohns
                xvt.append(0)
            elif("coli" in features[x].lower()): #if they have colitis
                xvt.append(1)
            else:
                xvt.append(-1) #if they have other
        if(x==7):
            try:
                xvt.append((2019-float(features[x]))/float(age)) #how long they've had the disease over their age
                #print((2019-float(features[x]))/float(age))
            except:
                xvt.append(0)
            #print((2019-float(features[x]))/float(age))
    userDictX[username]=xvt
for user in userDictY:
    if(max(userDictY[user])>0):
        userDictY[user].append(0) #last category is if no drug works
    else:
        userDictY[user].append(1) #no drug works

for user in userDictY:
    Y.append(userDictY[user])
    X.append(userDictX[user])

print('finished Processing')
np.random.seed(1)
randomize = list()
for ex in range(len(X)):
    randomize.append((X[ex],Y[ex]))
np.random.shuffle(randomize)

X =list()
Y = list()
for thing in randomize:
    X.append(thing[0])
    Y.append(thing[1])

Xtrain = X[:int(len(X)*.9)]#np.array(X)[:int(len(X)*.9)]
Ytrain = Y[:int(len(X)*.9)]#np.array(Y)[:int(len(X)*.9)]
y2train = list()
for thing in Ytrain:
    for x in range(len(thing)):
        if(thing[x]>0):
            y2train.append(x)
            break

Xtest = X[int(len(X)*.9):int(len(X)*.95)]#np.array(X)[int(len(X)*.9):int(len(X)*.95)]
Ytest = Y[int(len(X)*.9):int(len(X)*.95)]#np.array(Y)[int(len(X)*.9):int(len(X)*.95)]
y2test = list()
for thing in Ytest:
    for x in range(len(thing)):
        if(thing[x]>0):
            y2test.append(x)
            break

Xdev = X[int(len(X)*.95):]#np.array(X)[int(len(X)*.95):]
Ydev = Y[int(len(X)*.95):]#np.array(Y)[int(len(X)*.95):]
y2dev = list()
for thing in Ydev:
    for x in range(len(thing)):
        if(thing[x]>0):
            y2dev.append(x)
            break
y2train = np.array(y2train)
y2dev = np.array(y2dev)
y2test = np.array(y2test)

print('split into train, dev and test')

#print(Xtrain)
#print(Ytrain[0])
def makeDataset(index,dataX,dataY):
    dataset = list()
    for i in range(len(dataX)):
        thing = dataX[i]
        thing+=[dataY[i][index]]
        dataset.append(thing)
    return dataset


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0
 
# test predictions
'''weights = [-0.1, 0.20653640140000007, -0.23418117710000003, .1,0.1,0.1,0.1,0.1,0.1]
total = 0
correct = 0

for row in dataset:
    prediction = predict(row, weights)
    #print("Expected=%d, Predicted=%d" % (row[-1], prediction))
    total+=1
    correct += (row[-1] == prediction)

print(correct/float(total))'''

def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        #print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return weights

def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return(predictions)


n_folds = 3
l_rate = 0.01
n_epoch = 500
print('Testing...')
def test(nf,lr,ne,drug):
    dataset = makeDataset(drug,X,Y)
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    print('For drug: ' + str(drugNames[drug]))
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
    print('')

for x in range(len(Y[0])):
    test(n_folds,l_rate,n_epoch,x)