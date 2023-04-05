# This part is to decide the indices which are highly relative with the price prediction


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', None)


df = pd.read_csv('stock_data_1.csv', index_col=0)
df.drop(columns=['high', 'low', 'pre_close', 'Ln', 'Hn'], axis=1, inplace=True)
df.drop(columns=['turnover', 'range'], axis=1, inplace=True)
df.drop(columns=['Ki', 'Ji', 'Di', 'WR'], axis=1, inplace=True)
df.drop(columns=['DIF_UP', 'DIF_DN'], axis=1, inplace=True)
df.drop(columns=['MA', 'BBI'], axis=1, inplace=True)
# print(df.describe())
# scatter_matrix((df[['close', 'high', 'low', 'pre_close', 'Ln', 'Hn']]))
# scatter_matrix(df[['close','Ki', 'Ji', 'Di', 'RSV', 'WR']])
# scatter_matrix(df[['close', 'DIF','DIF_UP', 'DIF_DN', 'MA', 'BBI']], figsize=(8, 6))
# scatter_matrix(df[['MA', 'BBI']])
# plt.show()
# df.to_csv('xiaofei_total_pro_2.csv')
df.drop(columns=['change5', 'change10', 'change20'], axis=1, inplace=True)
df.drop(columns=['DIF', 'BIAS', 'rsi', 'DMA'], axis=1, inplace=True)

print(df.info())
# show correlation coefficient of variables
cov = np.corrcoef(df.T)
plt.figure(figsize=(8, 6))
# img = plt.imshow(cov)
label = ['close', 'change', 'vol', 'RSV', 'SD', ' ACD']
# img.set_xticklabels(label, fontsize=10)
# img.set_yticklabels(label, fontsize=10)
plt.title("coefficient of variables",fontsize=16)
# plt.colorbar(img, ticks=[0, 1])

sns.heatmap(cov, xticklabels=label, yticklabels=label, annot=True,
            cmap=sns.diverging_palette(10, 220, sep=80, n=7))
plt.show()