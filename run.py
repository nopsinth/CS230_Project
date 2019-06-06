import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from LogisticRegression import RegressionModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import InputLayer, Input, Masking
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from LSTM_classifier import RNNModel
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
    es = EarlyStopping(monitor='loss', mode = 'min', verbose = 1)
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

    model_rnn_multi_lstm = RNNModel()
    model_rnn_single_lstm = RNNModel()
    model_rnn_gru = RNNModel()
    model_rnn_lstm_gru = RNNModel()
    model_rnn_multi_lstm.train_multi_lstm(trainX, trainY)
    model_rnn_single_lstm.train_single_lstm(trainX, trainY)
    model_rnn_gru.train_gru(trainX, trainY)
    model_rnn_lstm_gru.train_lstm_gru(trainX, trainY)

    '''
    Grid Search (Tuning Hyperparameters)
    '''
    # model = KerasClassifier(build_fn = grid_model, epochs = 20, batch_size=32, verbose=0)
    # batch_size = [16, 32, 64, 128]
    # epochs = [10, 50, 100]
    # optimizers = ['rmsprop', 'adam']
    # learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    # momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    # param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer = optimizers)
    # grid = GridSearchCV(estimator = model, param_grid = param_grid)
    # grid_result = grid.fit(testX, testY)

    # print(grid_result.best_params_)
    '''
    Prediction
    '''
    y_pred = model_rnn_multi_lstm.predict(testX)
    acc = model_rnn_multi_lstm.evaluate(testX, testY)
    print("Acuuracy" + str(-acc))

    y_true = data['Ytest'][look_back+1:].ravel()
    print('AUC Score of', 'Multi-layer LSTM')
    print(roc_auc_score(np.array(y_true), np.array(y_pred.ravel())))

    y_pred = model_rnn_single_lstm.predict(testX)
    acc = model_rnn_single_lstm.evaluate(testX, testY)
    print("Acuuracy (LSTM)" + str(-acc))

    y_true = data['Ytest'][look_back+1:].ravel()
    print('AUC Score of', 'Single-layer LSTM')
    print(roc_auc_score(np.array(y_true), np.array(y_pred.ravel())))

    y_pred = model_rnn_gru.predict(testX)
    acc = model_rnn_gru.evaluate(testX, testY)
    print("Acuuracy (LSTM)" + str(-acc))

    y_true = data['Ytest'][look_back+1:].ravel()
    print('AUC Score of', 'GRU')
    print(roc_auc_score(np.array(y_true), np.array(y_pred.ravel())))

    y_pred = model_rnn_lstm_gru.predict(testX)
    acc = model_rnn_lstm_gru.evaluate(testX, testY)
    print("Acuuracy (LSTM)" + str(-acc))

    y_true = data['Ytest'][look_back+1:].ravel()
    print('AUC Score of', 'stacked LSTM and GRU')
    print(roc_auc_score(np.array(y_true), np.array(y_pred.ravel())))

    '''
    Prediction (Baseline)
    '''
    model_logistic = RegressionModel(1)
    model_ridge = RegressionModel(2)
    model_lasso = RegressionModel(3)
    model_logistic.train(data['Xtrain'], data['Ytrain'])
    model_ridge.train(data['Xtrain'], data['Ytrain'])
    model_lasso.train(data['Xtrain'], data['Ytrain'])

    Y_pred_logistic = model_logistic.predict(data['Xtest'])
    Y_pred_ridge = model_ridge.predict(data['Xtest'])
    Y_pred_lasso = model_lasso.predict(data['Xtest'])

    '''
    Evaluation
    '''
    #AUC score
    print('AUC Score of', 'Logistic')
    print(roc_auc_score(np.array(data['Ytest'][look_back+1:]), np.array(Y_pred_logistic[look_back+1:].ravel())))
    print('AUC Score of', 'Ridge')
    print(roc_auc_score(np.array(data['Ytest'][look_back+1:]), np.array(Y_pred_ridge[look_back+1:].ravel())))
    print('AUC Score of', 'Lasso')
    print(roc_auc_score(np.array(data['Ytest'][look_back+1:]), np.array(Y_pred_lasso[look_back+1:].ravel())))

    # Accuracy
    Y_pred_logistic_label = np.array(Y_pred_logistic > 0.5)
    Y_pred_ridge_label = np.array(Y_pred_ridge > 0.5)
    Y_pred_lasso_label = np.array(Y_pred_lasso > 0.5)
    print(accuracy_score(Ytest, Y_pred, normalize = True))
    print(accuracy_score(Ytest, Y_pred_ridge_label, normalize = True))
    print(accuracy_score(Ytest, Y_pred_lasso_label, normalize = True))

    # #F1-score
    # print(f1_score(Ytest, np.array(Y_pred), average='weighted'))
    # print(f1_score(Ytest, np.array(Y_pred_ridge > threshold), average='weighted'))
    # print(f1_score(Ytest, np.array(Y_pred_lasso > threshold), average='weighted'))

if __name__ == "__main__":
    main()
