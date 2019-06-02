import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from LogisticRegression import RegressionModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from LSTM_classifier import LSTMModel
import warnings

def get_data_by_years(df, years, features, label):
    ydf = df[df.Date.map(lambda x: x.year in years)]
    return ydf[features].values, ydf[label].values

def get_data(df, data_start, valid_start, test_start, data_end, features, label):
    data = {}
    train_years = list(range(data_start,  valid_start))
    valid_years = list(range(valid_start, test_start))
    test_years  = list(range(test_start,  data_end))
    data['Xtrain'], data['Ytrain'] = get_data_by_years(df, train_years, features, label)
    data['Xvalid'], data['Yvalid'] = get_data_by_years(df, valid_years, features, label)
    data['Xtest'],  data['Ytest']  = get_data_by_years(df, test_years,  features, label)
    return data

look_back = 10
def create_dataset(Xtrain, Ytrain, look_back):
    dataX, dataY = [], []
    for i in range(len(Xtrain)-look_back-1):
        a = Xtrain[i:(i+look_back)]
        dataX.append(a)
        dataY.append(Ytrain[i + look_back])
    return np.array(dataX), np.array(dataY)

num_features = 15
def grid_model(optimizer = 'rmsprop'):
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(look_back, num_features), return_sequences = True))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, return_sequences = True))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    es = EarlyStopping(monitor='loss', mode = 'min', verbose=1)
    # model.fit(X, Y, epochs = 20, batch_size = 64)
    return model


def main():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    df = pd.read_csv("mod_QQQ.csv")
    df['Date'] = pd.to_datetime(df["Date"], dayfirst = True)
    data_start = 1999
    valid_start = 2014
    test_start = 2015
    data_end = 2018
    features = [
        'EMA10',
        'EMA100',
        'EMA12',
        'EMA20',
        'EMA26',
        'EMA50',
        'SMA10',
        'SMA100',
        'SMA15',
        'SMA20',
        'SMA5',
        'SMA50',
        'MACD',
        'Close',
        'Volume'
    ]

    df.drop(["Unnamed: 0"],axis='columns', inplace=True)
    Next_PriceUp = df['PriceUp'][1:].values
    df = df[:-1]
    df = df.assign(Next_PriceUp = Next_PriceUp)
    label = ['Next_PriceUp']
    data = get_data(df, data_start, valid_start, test_start, data_end, features, label)

    scale = MinMaxScaler(feature_range=(0, 1))
    data['Xtrain'] = scale.fit_transform(data['Xtrain'])
    data['Ytrain'] = scale.fit_transform(data['Ytrain'].reshape(-1,1))
    data['Xtest'] = scale.fit_transform(data['Xtest'])
    data['Ytest'] = scale.fit_transform(data['Ytest'].reshape(-1,1))
    trainX, trainY = create_dataset(data['Xtrain'], data['Ytrain'], look_back)
    testX, testY = create_dataset(data['Xtest'], data['Ytest'], look_back)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    #lstm = LSTMModel()
    model = KerasClassifier(build_fn = grid_model, epochs = 20, batch_size=32, verbose=0)
    batch_size = [16, 32, 64, 128]
    epochs = [10, 50, 100]
    optimizers = ['rmsprop', 'adam']
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer = optimizers)
    grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = -1)
    grid_result = grid.fit(testX, testY)

    print(grid_result.best_params_)
    # print("Summarize the results")
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    #best_run, best_model = optim.minimize(model = create_model, data = data_func, algo = tpe.suggest, max_evals = 5, trials=Trials())
    #X_train, Y_train, X_test, Y_test = data_func()

    # y_pred = model.predict(testX)
    # acc = model.evaluate(testX, testY)

    # Xscaled = preprocessing.scale(Xtrain)
    # model = RegressionModel(1)
    # model_ridge = RegressionModel(2)
    # model_lasso = RegressionModel(3)
    # model.train(Xscaled, Ytrain)
    # model_ridge.train(Xscaled, Ytrain)
    # model_lasso.train(Xscaled, Ytrain)
    #
    # #Evaluation
    # #Accuracy
    # threshold = 0.5
    # Y_pred_ridge = np.array(Y_pred_ridge > 0.5)
    # Y_pred_lasso = np.array(Y_pred_lasso > 0.5)
    # print(accuracy_score(Ytest, Y_pred, normalize = True))
    # print(accuracy_score(Ytest, Y_pred_ridge, normalize = True))
    # print(accuracy_score(Ytest, Y_pred_lasso, normalize = True))
    #
    # #F1-score
    # print(f1_score(Ytest, np.array(Y_pred), average='weighted'))
    # print(f1_score(Ytest, np.array(Y_pred_ridge > threshold), average='weighted'))
    # print(f1_score(Ytest, np.array(Y_pred_lasso > threshold), average='weighted'))

if __name__ == "__main__":
    main()
