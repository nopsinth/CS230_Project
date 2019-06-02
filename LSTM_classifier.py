import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from hyperas import optim
# from hyperas.distributions import choice, uniform
# from hyperopt import Trials, STATUS_OK, tpe
from keras import optimizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class LSTMModel():

    def train(self, X, Y):
        assert len(X) == len(Y)
        self.model = Sequential()
        self.model.add(LSTM(units=128, input_shape=(X.shape[1], X.shape[2]), return_sequences = True))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(LSTM(units=128, return_sequences = True))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.7))
        self.model.add(LSTM(units=128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 1))
        self.model.add(Activation('sigmoid'))
        self.model.summary()
        #Pre-training
        #L2-regularization
        #Learning rate (Log scale)
        #Number of units and Number of epochs
        #{{choice([adam, 'adagrad', 'sgd'])}}
        optimizer = optimizers.Adam(lr = learn_rate, momentum = momentum)
        self.model.compile(loss = 'binary_crossentropy', optimizer = 'adagrad', metrics = ['accuracy'])
        es = EarlyStopping(monitor='loss', mode = 'min', verbose=1)
        # self.model.fit(X, Y, epochs = 20, batch_size = 64)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, Y):
        score, acc = self.model.evaluate(X, Y, verbose = 1)
        return -acc
