import keras.regularizers
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.saving.save import load_model
from numpy import float32, concatenate
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from process_data import get_processed_data

pd.set_option('display.width', 1000)


def pre_process(n: int):  # n means predict next n-day change(1,5,10,20)
    # stock_data = get_processed_data()
    stock_data = pd.read_csv('stock_data.csv')
    stock_data.set_index(['date'], inplace=True)
    if n not in [1, 5, 10, 20]:
        print('Only accept n = 1, 5, 10, 20, please re-enter!')
    elif n == 1:
        stock_data.drop(columns=['change5', 'change10', 'change20'], axis=1, inplace=True)
    elif n == 5:
        stock_data.drop(columns=['change1', 'change10', 'change20'], axis=1, inplace=True)
    elif n == 10:
        stock_data.drop(columns=['change1', 'change5', 'change20'], axis=1, inplace=True)
    elif n == 20:
        stock_data.drop(columns=['change1', 'change5', 'change10'], axis=1, inplace=True)
    values = stock_data.values
    values = values.astype(float32)
    return values


# normalization and reshape
def normalize_data(n: int):
    values = pre_process(n)
    sep = int(0.9 * len(values))
    train = values[:sep, :]
    test = values[sep:, :]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_fit = scaler.fit(train)
    train = scaler_fit.transform(train)
    test = scaler_fit.transform(test)
    return train, test


# get scaler_fit
def get_scaler_fit(n: int):
    values = pre_process(n)
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
def set_data(n: int):  # different strategy between n=1 or not
    train, test = normalize_data(n)

    # merge normalized data
    df = concatenate((train, test), axis=0)
    df = pd.DataFrame(df)
    data = df.values
    data = data.astype(float32)

    # reframe data to supervised
    reframed = series_to_supervised(data, 1, 1)
    reframed.drop(reframed.columns[[6, 7, 8, 9, 10, 11, 12]], axis=1, inplace=True)

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
def fit_model(n, m):    # m = parameters
    train_X, test_X, train_Y, test_Y = set_data(n)
    model = Sequential()
    model.add(GRU(50, dropout=0.1, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]),
                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.4))
    model.add(LSTM(30, dropout=0.1, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(20, dropout=0.1))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    model.fit(train_X, train_Y, epochs=25, batch_size=40, validation_split=0.2, verbose=0, shuffle=False,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
    loss = model.evaluate(test_X, test_Y)
    return loss


# test model with different parameters
def adjust_params(n):
    # define scope of search
    # params = [10, 15, 20, 25, 30, 50, 80]
    # params = [40, 60, 80, 100, 120, 150]
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # params = ['adam', 'AdaGrad', 'Adadelta', 'RMSProp']
    # params = ['sigmoid', 'softsign', 'tanh', 'selu']
    n_repeats = 5

    # grid search parameter values
    scores = pd.DataFrame()
    for value in params:
        # repeat each experiment multiple times
        loss_values = list()
        for i in range(n_repeats):
            loss = fit_model(n, value)
            loss_values.append(loss)
            print('>%d/%d param=%s, loss=%f' % (i + 1, n_repeats, value, loss))
        # store results for this parameter
        scores[str(value)] = loss_values
        # summary statistics of results
    print(scores.describe())
    # box and whisker plot of results
    plt.figure(figsize=(12, 6), dpi=100)
    scores.boxplot(ax=plt.gca())
    plt.title('Comparason of different dropout_2 in model mixed', fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('dropout_2', fontsize='10')
    plt.show()


# create LSTM model
def model_lstm(n: int):  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data(n)
    model = Sequential()
    # time_step = 1(train_X.shape[1]
    model.add(LSTM(60, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    # model.add(keras.layers.BatchNormalization())
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_lstm = model.fit(train_X, train_Y, epochs=15, batch_size=40, validation_split=0.2, verbose=0,
                             shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
    model.save('model/lstm_%d.h5' % n)
    return history_lstm


# create stacked LSTM model
def model_stacked_lstm(n: int):  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data(n)
    model = Sequential()
    model.add(LSTM(40, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(40))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_stacked_lstm = model.fit(train_X, train_Y, epochs=20, batch_size=60,
                                     validation_split=0.2, verbose=0, shuffle=False,
                                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
    model.save('model/stacked_lstm_%d.h5' % n)
    return history_stacked_lstm


# create GRU model
def model_gru(n: int):  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data(n)
    model = Sequential()
    model.add(GRU(70, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_stacked_gru = model.fit(train_X, train_Y, epochs=20, batch_size=40, validation_split=0.2,
                                    verbose=0, shuffle=False,
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
    model.save('model/gru_%d.h5' % n)
    return history_stacked_gru


# create stacked GRU model
def model_stacked_gru(n: int):  # to predict next n-day change
    train_X, test_X, train_Y, test_Y = set_data(n)
    model = Sequential()
    model.add(GRU(30, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.3))
    model.add(GRU(20))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history_gru = model.fit(train_X, train_Y, epochs=25, batch_size=40, validation_split=0.2, verbose=0,
                            shuffle=False, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
    model.save('model/stacked_gru_%d.h5' % n)
    return history_gru


# create mixed_model
def model_mixed(n: int):
    train_X, test_X, train_Y, test_Y = set_data(n)
    model = Sequential()
    model.add(GRU(60, dropout=0.1, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]),
                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(LSTM(40, dropout=0.1, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(GRU(30))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='softsign'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='auto')
    history_gru = model.fit(train_X, train_Y, epochs=200, batch_size=32, validation_split=0.2, verbose=0,
                            shuffle=False, callbacks=[reduce_lr,
                                                      EarlyStopping(monitor='val_loss', patience=5, verbose=0)])
    model.save('model/mixed_%d.h5' % n)
    print(model.summary())
    return history_gru


# show the loss
def show_loss(model_name, n: int):
    history = model_name(n)
    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('model_loss', fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('epoch', fontsize='10')
    plt.legend()
    # plt.show()


# prediction
def predict_result(model_name: str, n: int):  # import model_name
    model = load_model('model/' + model_name + '_' + str(n) + ".h5")
    train_X, test_X, train_Y, test_Y = set_data(n)
    y_predict = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # inverse normalization
    inv_y_test = concatenate((test_X[:, :6], y_predict), axis=1)
    scaler_fit = get_scaler_fit(n)
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
    latest_prediction = list()
    for i in [1, 5, 10, 20]:
        train, test = normalize_data(i)
        model = load_model('model/' + model_name + '_' + str(i) + ".h5")
        latest_data = test[:, :-1]

        latest_data = latest_data.reshape(latest_data.shape[0], 1, latest_data.shape[1])
        latest_pred = model.predict(latest_data)
        latest_data = latest_data.reshape((latest_data.shape[0], latest_data.shape[2]))

        # inverse normalization
        inv_latest = concatenate((latest_data[:, :6], latest_pred), axis=1)
        scaler_fit = get_scaler_fit(i)
        inv_latest = scaler_fit.inverse_transform(inv_latest)
        inv_latest_pred = inv_latest[:, -1]
        latest_prediction.append(inv_latest_pred[-1] * 100)
    return latest_prediction


# show prediction figure
def show_prediction(model_name: str, n: int):
    inv_y_true, inv_y_predict = predict_result(model_name, n)
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(inv_y_true, color='red', label='Original')
    plt.plot(inv_y_predict, color='green', label='Predict')
    plt.xlabel('the number of test data')
    plt.ylabel('change')
    plt.title('Prediction of ' + model_name)
    plt.legend()
    # plt.show()


# calculate error
def err_cal(model_name: str, n: int):
    inv_y_true, inv_y_predict = predict_result(model_name, n)
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
def show_prob_acc(model_name: str, n: int, err: float):
    inv_y_true, inv_y_predict = predict_result(model_name, n)
    accuracy = get_accuracy(inv_y_true, inv_y_predict, err)
    probability = get_probability(inv_y_true, inv_y_predict)

    return probability, accuracy


# show today's prediction
def show_latest_prediction(model_name: str):
    latest_prediction = predict_latest(model_name)
    prob_1, acc_1 = show_prob_acc(model_name, 1, 0.005)
    prob_5, acc_5 = show_prob_acc(model_name, 5, 0.01)
    prob_10, acc_10 = show_prob_acc(model_name, 10, 0.015)
    prob_20, acc_20 = show_prob_acc(model_name, 20, 0.02)
    probabilty = [prob_1, prob_5, prob_10, prob_20]
    accuracy = [acc_1, acc_5, acc_10, acc_20]
    days = [1, 5, 10, 20]
    for i in range(4):
        day = days[i]
        print('The next %d-day prediction is: %0.2f' % (day, latest_prediction[i]), '%', end='')
        print(' ', 'probability = %0.2f' % probabilty[i], '%', end='')
        print(' ', 'accuracy = %0.2f' % accuracy[i], '%')


# test performance of models
def test_performance(model_name: str):
    err_range = [0.005, 0.01, 0.02, 0.03]
    days = [1, 5, 10, 20]
    performance = {}
    for n in days:
        probability = []
        accuracy = []
        l_mae = []
        l_rmse = []
        l_rsquare = []
        for i in range(20):
            prob, acc = show_prob_acc(model_name, n, err_range[days.index(n)])
            probability.append(prob)
            accuracy.append(acc)
            mae, rmse, r_square = err_cal(model_name, n)
            l_mae.append(mae)
            l_rmse.append(rmse)
            l_rsquare.append(r_square)
        performance[n] = [np.mean(l_mae),
                          np.mean(l_rmse),
                          np.mean(l_rsquare),
                          np.mean(probability),
                          np.mean(accuracy)]
    return performance


# show performance of all models
def show_performance():
    lstm_perf = test_performance('lstm')
    stacked_lstm_perf = test_performance('stacked_lstm')
    gru_perf = test_performance('gru')
    stacked_gru_perf = test_performance('lstm')

    days_perf = []
    para_name = ['mae', 'rmse', 'square', 'probability', 'accuracy']
    for i in [1, 5, 10, 20]:
        days_perf.append({'lstm_'+str(i): lstm_perf[i],
                          'stacked_lstm_'+str(i): stacked_lstm_perf[i],
                          'gru_'+str(i): gru_perf[i],
                          'stacked_gru_'+str(i): stacked_gru_perf[i],
                          'param_name': para_name})
    df1 = pd.DataFrame(days_perf[0])
    df1.set_index(['param_name'], inplace=True)
    df2 = pd.DataFrame(days_perf[1])
    df2.set_index(['param_name'], inplace=True)
    df3 = pd.DataFrame(days_perf[2])
    df3.set_index(['param_name'], inplace=True)
    df4 = pd.DataFrame(days_perf[3])
    df4.set_index(['param_name'], inplace=True)
    df1.to_csv('day1_perf.csv')
    df2.to_csv('day5_perf.csv')
    df3.to_csv('day10_perf.csv')
    df4.to_csv('day20_perf.csv')
    return df1, df2, df3, df4


# single test
def single_test(model_name: str, n: int):
    probability, accuracy = show_prob_acc(model_name, n, 0.02)
    print('Probability to predict the right trend in %s: %.2f' % (model_name, probability), '%')
    print('Prediction accuracy of %s：%.2f' % (model_name, accuracy), '%')
    show_prediction(model_name, n)
    mae, rmse, r_square = err_cal(model_name, n)
    print('Sqrt of mean square error: %.4f' % rmse)
    print('Mean absolute error: %.4f' % mae)
    print('R_square: %.4f' % r_square)
    plt.show()


if __name__ == '__main__':
    # model_lstm(1)
    # model_lstm(5)
    # model_lstm(10)
    # model_lstm(20)
    model_stacked_lstm(1)
    # model_stacked_lstm(5)
    # model_stacked_lstm(10)
    # model_stacked_lstm(20)
    # model_gru(1)
    # model_gru(5)
    # model_gru(10)
    model_gru(20)
    # model_stacked_gru(1)
    # model_stacked_gru(5)
    # model_stacked_gru(10)
    # model_stacked_gru(20)
    # model_mixed(20)
    show_loss(model_gru, 20)
    single_test('gru', 20)

    # adjust_params(20)

# lstm_para = pd.DataFrame({#
#         'epoch':25,
#         'batch_size':100,
#         'units':32,
#         'dropout':0.5,
#         'activation':'relu',
#         'optimizer':'adam'
#     }, index=['lstm1'])
# para_row = {
#         'epoch':50,
#         'batch_size':70,
#         'units':40,
#         'dropout':0.3
#         'activation':'relu',
#         'optimizer':'adam'
#     }
# lstm_result = pd.DataFrame({
#     'MSE':16.211256,
#     'RMSE':4.02632,
#     'MAE':3.120912,
#     'R_squre':0.833925,
#     'probability':92.494929,
#     'accuracy':42.799189,}, index=['lstm1'])
# result_row = {
#     'MSE':mse,
#     'RMSE':rmse,
#     'MAE':mae,
#     'R_squre':r_square,
#     'probability':prob,
#     'accuracy':accuracy,}
# lstm_para.loc['lstm4'] = para_row
# lstm_result.loc['lstm4'] = result_row
# print(lstm_result)
# print(lstm_para)
