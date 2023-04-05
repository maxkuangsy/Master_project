# choose single GRU model to predict the next 20-day change

import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
from numpy import float32, concatenate
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from process_data import get_processed_data, get_latest_data


def pre_process(code: str):
    stock_data = pd.read_csv('data/train_data_' + code + '.csv')
    stock_data.set_index(['date'], inplace=True)
    stock_data.drop(columns=['change1', 'change5', 'change10'], axis=1, inplace=True)
    values = stock_data.values
    values = values.astype(float32)
    return values


# normalization and reshape
def normalize_data(code: str):
    values = pre_process(code)
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    test = values[sep:, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_fit = scaler.fit(train)
    train = scaler_fit.transform(train)
    test = scaler_fit.transform(test)
    return train, test


# process latest data
def normalize_latest_data(code: str):
    stock_data = pd.read_csv('data/latest_data_' + code + '.csv')
    stock_data.set_index(['date'], inplace=True)
    stock_data.drop(columns=['change1', 'change5', 'change10'], axis=1, inplace=True)
    values = stock_data.values
    values = values.astype(float32)
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    test = values[sep:, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_fit = scaler.fit(train)
    train = scaler_fit.transform(train)
    test = scaler_fit.transform(test)
    return train, test, scaler_fit


# get scaler_fit
def get_scaler_fit(code: str):
    values = pre_process(code)
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_fit = scaler.fit(train)
    return scaler_fit


# change series to supervised data
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols = []
    names = []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# set train and test data, then normalization
def set_data(code: str):
    train, test = normalize_data(code)

    # merge normalized data
    df = concatenate((train, test), axis=0)
    df = pd.DataFrame(df)
    data = df.values
    data = data.astype(float32)

    # reframe data to supervised
    reframed = series_to_supervised(data, 1, 1)
    reframed.drop(reframed.columns[[7, 8, 9, 10, 11, 12]], axis=1, inplace=True)

    # split train and test data
    values = reframed.values
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    test = values[sep:, :]
    train_X, train_Y = train[:, : -1], train[:, -1]
    test_X, test_Y = test[:, : -1], test[:, -1]

    # reshape input to 3D
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, test_X, train_Y, test_Y


def get_probability(true, pred):
    count = 0
    for i in range(len(true)):
        if true[i] * pred[i] >= 0:
            count += 1
    prob = count / len(true)
    return prob * 100


def get_accuracy(true, pred, n):
    count = 0
    for i in range(len(true)):
        if abs(true[i] - pred[i]) <= n:
            count += 1
    accuracy = count / len(true)
    return accuracy * 100


# # available for being imported(gru model)
def predictor(code: str):  # to predict next n-day change
    change_20 = []  # next 20-day change
    probability = []
    accuracy = []
    performance = []
    for i in range(10):
        train_X, test_X, train_Y, test_Y = set_data(code)
        model = Sequential()
        model.add(GRU(50, dropout=0.04, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dropout(0.18))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mae', optimizer='adam')
        # fit network
        model.fit(train_X, train_Y, epochs=25, batch_size=20, validation_split=0.2, verbose=0, shuffle=False,
                  callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                   patience=6, verbose=0, mode='min')])

        # predict the test data
        train_X, test_X, train_Y, test_Y = set_data(code)
        y_predict = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        # inverse normalization
        inv_y_test = concatenate((test_X[:, :6], y_predict), axis=1)
        scaler_fit = get_scaler_fit(code)
        inv_y_test = scaler_fit.inverse_transform(inv_y_test)
        inv_y_predict = inv_y_test[:, -1]

        # invert scaling for actual
        test_Y = test_Y.reshape((len(test_Y), 1))
        inv_true = concatenate((test_X[:, :6], test_Y), axis=1)
        inv_true = scaler_fit.inverse_transform(inv_true)
        inv_y_true = inv_true[:, -1]

        # get prob and acc
        acc = get_accuracy(inv_y_true, inv_y_predict, 0.02)
        prob = get_probability(inv_y_true, inv_y_predict)
        accuracy.append(acc)
        probability.append(prob)

        # get today's prediction
        train, test, scaler_fit2 = normalize_latest_data(code)
        latest_data = test[:, :]

        latest_data = latest_data.reshape(latest_data.shape[0], 1, latest_data.shape[1])
        latest_pred = model.predict(latest_data)
        latest_data = latest_data.reshape((latest_data.shape[0], latest_data.shape[2]))

        # inverse normalization
        inv_latest = concatenate((latest_data[:, :6], latest_pred), axis=1)
        inv_latest = scaler_fit2.inverse_transform(inv_latest)
        inv_latest_pred = inv_latest[:, -1]
        latest_prediction = inv_latest_pred[-1] * 100
        change_20.append(latest_prediction)

    # calculate mean value
    performance.append(np.mean(change_20))
    performance.append(np.mean(probability))
    performance.append(np.mean(accuracy))

    print('The next 20-day prediction of %s is: %0.2f' % (dic[code], performance[0]), '%', end='')
    print(' ', 'probability = %0.2f' % performance[1], '%', end='')
    print(' ', 'accuracy = %0.2f' % performance[2], '%')


# get data
def get_data(code: str):
    df1 = get_processed_data(code)
    df2 = get_latest_data(code)
    df1.to_csv('data/train_data_' + code + '.csv', index=False)
    df2.to_csv('data/latest_data_' + code + '.csv', index=False)


if __name__ == '__main__':
    dic = {}     # create dictionary of code and name
    dic['0000934'] = 'jinrong'
    dic['0000808'] = 'yiyao'
    dic['0000067'] = 'xinxinghunhe'
    dic['0000807'] = 'shipinyinliao'
    dic['0000993'] = 'xinxi'
    # list = ['0000934', '0000808', '0000067', '0000807', '0000993']
    # for i in range(5):
    #     code = list[i]
    #     get_data(code)
    #     predictor(code)

    predictor('0000934')