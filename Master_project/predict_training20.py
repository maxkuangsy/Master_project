import keras.regularizers
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.saving.save import load_model
from numpy import float32, concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from process_data import get_processed_data

pd.set_option('display.width', 1000)


def pre_process():
    # stock_data = get_processed_data('0000006')
    stock_data = pd.read_csv('stock_data.csv')  # get the processed data directly from the file due to the invalid downloading link
    stock_data.set_index(['date'], inplace=True)
    stock_data.drop(columns=['change1', 'change5', 'change10'], axis=1, inplace=True)   # omit 1-day, 5-day and 10-day prediction
    values = stock_data.values
    values = values.astype(float32)
    return values


# normalization and reshape
def normalize_data():
    values = pre_process()
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    test = values[sep:, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_fit = scaler.fit(train)
    train = scaler_fit.transform(train)
    test = scaler_fit.transform(test)
    return train, test


# get scaler_fit
def get_scaler_fit():
    values = pre_process()
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_fit = scaler.fit(train)
    return scaler_fit


# change series to supervised data
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
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
def set_data():
    train, test = normalize_data()

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


# create test_model
def fit_model(m):    # m = parameters
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    # time_step = 1(train_X.shape[1]
    model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2])))
    # lstm = LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2]))
    # model.add(Bidirectional(lstm))
    model.add(Dropout(m))
    # model.add(keras.layers.BatchNormalization())
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    model.fit(train_X, train_Y, epochs=12, batch_size=20, validation_split=0.2, verbose=0, shuffle=False,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=12, verbose=0, mode='min')])
    loss = model.evaluate(test_X, test_Y)
    return loss


# test model with different parameters
def adjust_params():
    # define scope of search
    # params = [15, 20, 25, 30, 40, 50]
    # params = [20, 32, 40, 50, 60, 80, 100, 120, 150]
    # params = [20, 21, 22, 23, 24, 25]
    # params = [20, 25, 30, 40, 50, 60]
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # params = [0.02, 0.04, 0.06, 0.08, 0.10]
    # params = ['adam', 'AdaGrad', 'Adadelta', 'RMSProp']
    # params = ['sigmoid', 'softsign', 'tanh', 'selu']
    n_repeats = 10

    # grid search parameter values
    scores = pd.DataFrame()
    for value in params:
        # repeat each experiment multiple times
        loss_values = list()
        for i in range(n_repeats):
            loss = fit_model(value)
            loss_values.append(loss)
            print('>%d/%d param=%s, loss=%f' % (i + 1, n_repeats, value, loss))
        # store results for this parameter
        scores[str(value)] = loss_values
        # summary statistics of results
    print(scores.describe())
    # box and whisker plot of results
    plt.figure(figsize=(12, 6), dpi=100)
    scores.boxplot(ax=plt.gca())
    plt.title('Comparason of different dropout in model LSTM', fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('dropout', fontsize='10')
    plt.show()


# create LSTM model
def model_lstm():  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    # time_step = 1(train_X.shape[1]
    model.add(LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2])))
    # lstm = LSTM(80, input_shape=(train_X.shape[1], train_X.shape[2]))
    # model.add(Bidirectional(lstm))
    model.add(Dropout(0.1))
    # model.add(keras.layers.BatchNormalization())
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_lstm = model.fit(train_X, train_Y, epochs=60, batch_size=35, validation_split=0.2, verbose=0,
                             shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                   patience=12, verbose=0, mode='min')])
    model.save('model/lstm.h5')
    return history_lstm


# create stacked LSTM model
def model_stacked_lstm():  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.4))
    model.add(LSTM(50))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_stacked_lstm = model.fit(train_X, train_Y, epochs=40, batch_size=70, validation_split=0.2, verbose=0,
                                     shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                                             patience=10, verbose=0, mode='min')])
    model.save('model/stacked_lstm.h5')
    return history_stacked_lstm


# create GRU model
def model_gru():  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    model.add(GRU(50, dropout=0.04, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.18))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')
    history_stacked_gru = model.fit(train_X, train_Y, epochs=25, batch_size=20, validation_split=0.2,
                                    verbose=0, shuffle=False, callbacks=
                                    [EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                   patience=6, verbose=0, mode='min')])
    model.save('model/gru.h5')
    return history_stacked_gru


# create stacked GRU model
def model_stacked_gru():  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.4))
    model.add(GRU(20))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_gru = model.fit(train_X, train_Y, epochs=40, batch_size=70, validation_split=0.2, verbose=0,
                            shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                   patience=10, verbose=0, mode='min')])
    model.save('model/stacked_gru.h5')
    return history_gru


def model_lstm_gru():
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.3))
    # model.add(keras.layers.BatchNormalization())
    model.add(GRU(20, kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='auto')
    history_mixed = model.fit(train_X, train_Y, epochs=25, batch_size=20, validation_split=0.2, verbose=0,
                              shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                   patience=6, verbose=0, mode='min')])
    model.save('model/lstm_gru.h5')
    return history_mixed


# create mixed_model
def model_mixed():
    train_X, test_X, train_Y, test_Y = set_data()
    model = Sequential()
    model.add(GRU(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]),
                   kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(Dropout(0.4))
    # model.add(keras.layers.BatchNormalization())
    model.add(GRU(20,  return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(30))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_mixed = model.fit(train_X, train_Y, epochs=24, batch_size=32, validation_split=0.2, verbose=0,
                              shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001,
                                                   patience=12, verbose=0, mode='min')])
    model.save('model/mixed.h5')
    return history_mixed


# show the loss
def show_loss(model_name):
    history = model_name()
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('model_loss', fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('epoch', fontsize='10')
    plt.legend()
    plt.show()


# prediction
def predict_result(model_name: str):  # import model_name
    model = load_model('model/' + model_name + ".h5")
    train_X, test_X, train_Y, test_Y = set_data()
    y_predict = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # inverse normalization
    inv_y_test = concatenate((test_X[:, :6], y_predict), axis=1)
    scaler_fit = get_scaler_fit()
    inv_y_test = scaler_fit.inverse_transform(inv_y_test)
    inv_y_predict = inv_y_test[:, -1]

    # invert scaling for actual
    test_Y = test_Y.reshape((len(test_Y), 1))
    inv_true = concatenate((test_X[:, :6], test_Y), axis=1)
    inv_true = scaler_fit.inverse_transform(inv_true)
    inv_y_true = inv_true[:, -1]
    return inv_y_true, inv_y_predict


# get the prediction today
def predict_latest(model_name: str):
    train, test = normalize_data()
    model = load_model('model/' + model_name + ".h5")
    latest_data = test[:, :]

    latest_data = latest_data.reshape(latest_data.shape[0], 1, latest_data.shape[1])
    latest_pred = model.predict(latest_data)
    latest_data = latest_data.reshape((latest_data.shape[0], latest_data.shape[2]))

    # inverse normalization
    inv_latest = concatenate((latest_data[:, :6], latest_pred), axis=1)
    scaler_fit = get_scaler_fit()
    inv_latest = scaler_fit.inverse_transform(inv_latest)
    inv_latest_pred = inv_latest[:, -1]
    latest_prediction = inv_latest_pred * 100
    return latest_prediction


# show prediction figure
def show_prediction(model_name: str):
    inv_y_true, inv_y_predict = predict_result(model_name)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(inv_y_true, color='red', label='Original')
    plt.plot(inv_y_predict, color='green', label='Predict')
    plt.xlabel('the number of test data')
    plt.ylabel('change')
    plt.title('Prediction of ' + model_name)
    plt.legend()
    # plt.show()


# calculate error
def err_cal(model_name: str):
    inv_y_true, inv_y_predict = predict_result(model_name)
    # calculate MSE
    # mse = mean_squared_error(inv_y_true, inv_y_predict)
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y_true, inv_y_predict))
    # calculate MAE
    mae = mean_absolute_error(inv_y_true, inv_y_predict)
    # calculate R square
    r_square = r2_score(inv_y_true, inv_y_predict)

    return mae, rmse, r_square


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


# show accuracy and probability
def show_prob_acc(model_name: str):
    inv_y_true, inv_y_predict = predict_result(model_name)
    accuracy = get_accuracy(inv_y_true, inv_y_predict, 0.02)
    probability = get_probability(inv_y_true, inv_y_predict)

    return probability, accuracy


# show today's prediction
def show_latest_prediction(model_name: str):
    latest_prediction = predict_latest(model_name)
    prob, acc = show_prob_acc(model_name)
    print('The next 20-day prediction is: %0.2f' % latest_prediction[-1], '%', end='')
    print(' ', 'probability = %0.2f' % prob, '%', end='')
    print(' ', 'accuracy = %0.2f' % acc, '%')


# test performance of models
def test_performance(model, model_name: str):
    performance = []
    probability = []
    accuracy = []
    l_mae = []
    l_rmse = []
    l_rsquare = []
    for i in range(20):
        model()
        prob, acc = show_prob_acc(model_name)
        probability.append(prob)
        accuracy.append(acc)
        mae, rmse, r_square = err_cal(model_name)
        l_mae.append(mae)
        l_rmse.append(rmse)
        l_rsquare.append(r_square)
    performance.append(np.mean(l_mae))
    performance.append(np.std(l_mae))
    performance.append(np.mean(l_rmse))
    performance.append(np.std(l_rmse))
    performance.append(np.mean(l_rsquare))
    performance.append(np.std(l_rsquare))
    performance.append(np.mean(probability))
    performance.append(np.std(probability))
    performance.append(np.mean(accuracy))
    performance.append(np.std(accuracy))
    return performance


# show performance of all models
def save_performance():
    lstm_perf = test_performance(model_lstm , 'lstm')
    stacked_lstm_perf = test_performance(model_stacked_lstm, 'stacked_lstm')
    gru_perf = test_performance(model_gru, 'gru')
    stacked_gru_perf = test_performance(model_stacked_gru, 'stacked_gru')
    mixed_perf = test_performance(model_mixed, 'mixed')

    para_name = ['mae', 'mae_std', 'rmse', 'rmse_std', 'square', 'square_std',
                 'probability', 'probability_std', 'accuracy', 'accuracy_std']
    perf_all = {'param_name': para_name,
                'lstm_': lstm_perf,
                'stacked_lstm_': stacked_lstm_perf,
                'gru_': gru_perf,
                'stacked_gru_': stacked_gru_perf,
                'mixed': mixed_perf}
    df = pd.DataFrame(perf_all)
    # df.set_index(['param_name'], inplace=True)
    df.to_csv('models_performance_2.csv', index=False)
    return df


# single test
def single_test(model_name: str):
    probability, accuracy = show_prob_acc(model_name)
    print('Probability to predict the right trend in %s: %.2f' % (model_name, probability), '%')
    print('Prediction accuracy of %s：%.2f' % (model_name, accuracy), '%')
    show_prediction(model_name)
    mae, rmse, r_square = err_cal(model_name)
    print('Sqrt of mean square error: %.4f' % rmse)
    print('Mean absolute error: %.4f' % mae)
    print('R_square: %.4f' % r_square)
    plt.show()


def mul_test(model, model_name: str):
    result = test_performance(model, model_name)
    mae = result[0]
    mse_std = result[1]
    rmse = result[2]
    r_square = result[4]
    probability = result[6]
    accuracy = result[8]
    print('Probability to predict the right trend in %s: %.2f' % (model_name, probability), '%')
    print('Prediction accuracy of %s：%.2f' % (model_name, accuracy), '%')
    print('Sqrt of mean square error: %.4f' % rmse)
    print('Mean absolute error: %.4f' % mae)
    print('R_square: %.4f' % r_square)
    print('Std of Mean absolute error: %.4f' % mse_std)


def plt_show(model, model_name: str):
    df = get_processed_data('0000934')
    df.to_csv('stock_data.csv', index=False)
    model()
    inv_y_true1, inv_y_predict1 = predict_result(model_name)
    plt.figure(figsize=(12, 6), dpi=100)
    fig1 = plt.subplot(1, 2, 1)
    plt.plot(inv_y_true1, color='red', label='Original')
    plt.plot(inv_y_predict1, color='green', label='Predict')
    plt.xlabel('Number of test data')
    plt.ylabel('Change')
    plt.title('Prediction of 0000934 using ' + model_name)
    plt.legend()
    df = get_processed_data('0000006')
    df.to_csv('stock_data.csv', index=False)
    model()
    inv_y_true2, inv_y_predict2 = predict_result(model_name)
    fig2 = plt.subplot(1, 2, 2)
    plt.plot(inv_y_true2, color='red', label='Original')
    plt.plot(inv_y_predict2, color='green', label='Predict')
    plt.xlabel('Number of test data')
    plt.ylabel('Change')
    plt.title('Prediction of 0000006 using ' + model_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # model_gru()
    # show_loss(model_gru)
    # single_test('gru')
    # mul_test(model_gru, 'gru')

    # adjust_params()

    # plt_show(model_gru, 'gru')
    show_latest_prediction('gru')