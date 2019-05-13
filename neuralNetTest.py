import math
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import defaultdict
from textblob import TextBlob
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
for thing in open('userInfo.txt','r'):
    if(first):
        first=False
        continue
    info = thing.split('/t')
    features = info[2:9]
    username = info[0]
    Yval = info[10:]
    yvt = list()
    for thing in Yval:
        if(thing.lower()=='yes'):
            yvt.append(1)
        else:
            yvt.append(0)
    userDictY[username] = yvt
    xvt = list()
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
            xvt.append(2019-int(features[x])) #how long they've had the disease
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

learning_rate = 0.001
training_epochs = 100
batch_size = 100
display_step = 1

n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = 10 # Number of feature
n_classes = 2 # Number of classes to predict

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(X)/batch_size)
        X_batches = np.array_split(X, total_batch)
        Y_batches = np.array_split(Y, total_batch)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = X_batches[i], Y_batches[i]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: X_test, y: Y_test}))
    global result
    result = tf.argmax(pred, 1).eval({x: X_test, y: Y_test})