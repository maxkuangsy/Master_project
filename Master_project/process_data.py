import pandas as pd
from fractions import Fraction
from download_data import download_data
pd.set_option("display.max_columns", None)


def get_Ln(df, n):   # return the lowest price among previous n days
    buffer = []     # list which stores low data
    low = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 3])  # low_data is in column 3

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            lowest = min(buffer[i: i+n])
        else:
            lowest = 0
        low.append(lowest)

    df_low = pd.DataFrame({'Ln': pd.Series(low)})
    return df_low


def get_Hn(df, n):  # return the highest price among previous n days
    buffer = []      # list which stores high data
    high = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 2])  # high_data is in column 2

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            highest = max(buffer[i:i + n])
        else:
            highest = 0
        high.append(highest)

    df_high = pd.DataFrame({'Hn': pd.Series(high)})
    return df_high


def get_RSV(x):
    if (x['Hn'] - x['Ln']) == 0:
        return 0
    else:
        return (x['close'] - x['Ln']) * 100/(x['Hn'] - x['Ln'])


def get_Ki(df):
    buffer = []     # list which stores k value
    rsv = []
    for i in range(len(df)):
        rsv.append(df.iat[i, 11])  # rsv_data is in column 11
    rsv.reverse()   # to make the oldest data be the head
    buffer.append(50)   # first k_data is 50
    for i in range(len(df)-1):
        buffer.append(Fraction(2, 3)*buffer[i] + Fraction(1, 3)*rsv[i+1])
    # reverse back the list
    buffer.reverse()
    # create dataframe of k value
    df_k = pd.DataFrame({'Ki': pd.Series(buffer)})
    return df_k


def get_Di(df):
    buffer = []     # list which stores D value
    k = []
    for i in range(len(df)):
        k.append(df.iat[i, 12])  # k_data is in column 12
    k.reverse()   # to make the oldest data be the head
    buffer.append(50)   # first D_data is 50
    for i in range(len(df)-1):
        buffer.append(Fraction(2, 3)*buffer[i] + Fraction(1, 3)*k[i+1])
    # reverse back the list
    buffer.reverse()
    # create dataframe of k value
    df_d = pd.DataFrame({'Di': pd.Series(buffer)})
    return df_d


def get_Ji(x):
    return 3*x['Ki'] - 2*x['Di']


def get_MA(df, n: int):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            mean = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            mean = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            mean = buffer[i]
        ma.append(mean)

    df_ma = pd.DataFrame({'MA': pd.Series(ma)})
    return df_ma


def get_BIAS(x):
    return (x['close'] - x['MA'])/x['MA']


def get_SD(df, n):  # return the standard deviation price among previous n days
    close = []  # list which stores close data
    ma = []     # list which stores MA data
    sd = []
    for i in range(len(df)):
        close.append(df.iat[i, 1])  # close_data is in column 1
        ma.append(df.iat[i, 15])  # MA_data is in column 15

    for i in range(len(close)):
        summ = 0
        if i+n <= len(close)-1:
            for j in range(n):
                summ += (close[i+j] - ma[i+j])**2
            d = abs(summ / n) ** 0.5
            d = float(d)
            sd.append(d)
        else:
            d = 0
            d = float(d)
            sd.append(d)

    df_sd = pd.DataFrame({'SD': pd.Series(sd)})
    return df_sd


def get_WR(x):
    if (x['Hn'] - x['Ln']) == 0:
        return 0
    else:
        return (x['Hn'] - x['close']) * 100/(x['Hn'] - x['Ln'])


def get_DIF_UP(x):
    return x['close'] - min(x['low'], x['pre_close'])


def get_DIF_down(x):
    return x['close'] - max(x['high'], x['pre_close'])


def get_DIF(x):
    if x['change'] > 0:
        return x['DIF_UP']
    elif x['change'] < 0:
        return x['DIF_DN']
    elif x['change'] == 0:
        return 0


def get_ACD(df, n):  # return the acd among previous n days
    dif = []  # list which stores dif data
    acd = []    # list which stores acd data
    for i in range(len(df)):
        dif.append(df.iat[i, 21])  # dif_data is in column 21

    for i in range(len(dif)):
        summ = 0
        if i+n <= len(dif)-1:
            for j in range(n):
                summ += dif[i+j]
            acd.append(summ)
        else:
            acd.append(summ)

    df_acd = pd.DataFrame({'ACD': pd.Series(acd)})
    return df_acd


def get_RSI(df, n):  # return the RSI among previous n days
    change = []  # list which stores change data
    rsi = []    # list which stores rsi data
    for i in range(len(df)):
        change.append(df.iat[i, 6])  # change_data is in column6

    for i in range(len(change)):
        sum_up = 0
        sum_dn = 0
        buffer = 0
        if i+n <= len(change)-1:
            for j in range(n):
                if change[i+j] >= 0:
                    sum_up += change[i+j]
                else:
                    sum_dn += abs(change[i+j])
            if (sum_up + sum_dn) == 0:
                print('denomminator is 0. index = ', i + j)
            else:
                buffer = sum_up / (sum_up + sum_dn)
            rsi.append(buffer)
        else:
            for j in range(len(change)-i-1):
                if change[i+j] >= 0:
                    sum_up += change[i + j]
                else:
                    sum_dn += abs(change[i + j])
            if (sum_up + sum_dn) == 0:
                print('denomminator is 0. index = ', i+j)
            else:
                buffer = sum_up / (sum_up + sum_dn)
            rsi.append(buffer)

    df_rsi = pd.DataFrame({'rsi': pd.Series(rsi)})
    return df_rsi


def get_MA10(df, n):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            aver = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            aver = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            aver = buffer[i]
        ma.append(aver)

    df_ma = pd.DataFrame({'MA10': pd.Series(ma)})
    return df_ma


def get_MA50(df, n):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            mean = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            mean = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            mean = buffer[i]
        ma.append(mean)

    df_ma = pd.DataFrame({'MA50': pd.Series(ma)})
    return df_ma


def get_MA3(df, n):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            mean = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            mean = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            mean = buffer[i]
        ma.append(mean)

    df_ma = pd.DataFrame({'MA3': pd.Series(ma)})
    return df_ma


def get_MA6(df, n):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            mean = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            mean = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            mean = buffer[i]
        ma.append(mean)

    df_ma = pd.DataFrame({'MA6': pd.Series(ma)})
    return df_ma


def get_MA12(df, n):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            mean = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            mean = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            mean = buffer[i]
        ma.append(mean)

    df_ma = pd.DataFrame({'MA12': pd.Series(ma)})
    return df_ma


def get_MA24(df, n):  # return the mean price among previous n days
    buffer = []  # list which stores close data
    ma = []
    for i in range(len(df)):
        buffer.append(df.iat[i, 1])  # close_data is in column 1

    for i in range(len(buffer)):
        if i + n <= len(buffer) - 1:
            mean = sum(buffer[i: i + n])/n
        elif i < len(buffer) - 1:
            mean = sum(buffer[i:])/(len(buffer) - i - 1)
        elif i == len(buffer) - 1:
            mean = buffer[i]
        ma.append(mean)

    df_ma = pd.DataFrame({'MA24': pd.Series(ma)})
    return df_ma


def get_more_change(df, n):  # return the change among previous n days
    change = []  # list which stores change_oneday data
    buffer = []    # list which stores change_period data
    for i in range(len(df)):
        change.append(df.iat[i, 6])  # change_data is in column 6

    for i in range(len(change)):
        summ = 0
        if i < n-1:
            for j in range(i+1):
                summ += change[i-j]
            buffer.append(summ)
        else:
            for j in range(n):
                summ += change[i-j]
            buffer.append(summ)

    df_nchange = pd.DataFrame({'change%d' % n: pd.Series(buffer)})
    return df_nchange


def add_index(df):
    # add the column 'Hn' and 'Ln'
    df1 = pd.concat([df, get_Ln(df, 20), get_Hn(df, 20)], axis=1, join='outer')
    # change the 'change' to percentage data
    df1['change'] = df1.apply(lambda x: x['change']/100, axis=1)
    # add the column 'RSV'
    df1['RSV'] = df1.apply(get_RSV, axis=1, result_type='expand')
    # add the column 'Ki'
    df2 = pd.concat([df1, get_Ki(df1)], axis=1, join='outer')
    # add the colum 'DI'
    df3 = pd.concat([df2, get_Di(df2)], axis=1, join='outer')
    # add the column od 'Ji'
    df3['Ji'] = df3.apply(get_Ji, axis=1, result_type='expand')
    # add 'MA'
    df4 = pd.concat([df3, get_MA(df3, 20)], axis=1, join='outer')
    # add 'BIAS'
    df4['BIAS'] = df4.apply(get_BIAS, axis=1, result_type='expand')
    # add 'SD'
    df5 = pd.concat([df4, get_SD(df4, 20)], axis=1, join='outer')
    # add 'WR', 'DIF'
    df5['WR'] = df5.apply(get_WR, axis=1, result_type='expand')
    df5['DIF_UP'] = df5.apply(get_DIF_UP, axis=1, result_type='expand')
    df5['DIF_DN'] = df5.apply(get_DIF_down, axis=1, result_type='expand')
    df5['DIF'] = df5.apply(get_DIF, axis=1, result_type='expand')
    # add 'ACD'
    df6 = pd.concat([df5, get_ACD(df5, 20)], axis=1, join='outer')
    # add 'RSI'
    df7 = pd.concat([df6, get_RSI(df6, 20)], axis=1, join='outer')
    # add MA10, MA50
    df8 = pd.concat([df7, get_MA10(df7, 10), get_MA50(df7, 50)], axis=1, join='outer')
    # add DMA
    df8['DMA'] = df8.apply(lambda x: x['MA10'] - x['MA50'], axis=1, result_type='expand')
    # drop MA10, MA50
    df9 = df8.drop(columns=['MA10', 'MA50'], axis=1)
    # add MA3,6,12,24
    df10 = pd.concat([df9, get_MA3(df9, 3), get_MA6(df9, 6), get_MA12(df9, 12), get_MA24(df9, 24)],
                     axis=1, join='outer')
    # add BBI
    df10['BBI'] = df10.apply(lambda x: (x['MA3'] + x['MA6'] + x['MA12'] + x['MA24'])/4,
                             axis=1, result_type='expand')
    df11 = pd.concat([df10, get_more_change(df10, 5), get_more_change(df10, 10), get_more_change(df10, 20)],
                     axis=1, join='outer')
    # drop MA3,6,12,24
    df12 = df11.drop(columns=['MA3', 'MA6', 'MA12', 'MA24'], axis=1)

    return df12


def reverse_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)     # reverse the stock data by date
    return df


def drop_extra(df):     # drop extra indices which have low relevance with the result
    df.drop(columns=['high', 'low', 'pre_close', 'Ln', 'Hn'], axis=1, inplace=True)
    df.drop(columns=['turnover', 'range'], axis=1, inplace=True)
    df.drop(columns=['Ki', 'Ji', 'Di', 'WR'], axis=1, inplace=True)
    df.drop(columns=['DIF_UP', 'DIF_DN', 'DIF'], axis=1, inplace=True)
    df.drop(columns=['MA', 'BBI'], axis=1, inplace=True)
    d = df.pop('change')
    df.insert(9, 'change', d)
    df.insert(10, 'change1', d)
    df.drop([0], axis=0, inplace=True)
    df.drop(columns=['BIAS', 'rsi', 'DMA'], axis=1, inplace=True)
    return df


def get_processed_data(code: str):
    df = add_index(download_data(code))
    df = reverse_data(df)
    df = drop_extra(df)
    df = df.iloc[20: -20, : ] # delete the first 20 and last 20 lines
    return df


# get latest data
def get_latest_data(code: str):
    df = add_index(download_data(code))
    df = reverse_data(df)
    df = drop_extra(df)
    df = df.iloc[20: , :]  # delete the first 20 lines
    return df

# get the data we want and save
if __name__ == '__main__':
    new_df = get_processed_data('0000006')
    print(new_df.info())
    new_df.to_csv('stock_data.csv', index=False)
