import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import predict_training20
from process_data import get_processed_data
from download_data import download_data
from process_data import reverse_data
from predict_training20 import single_test
pd.set_option('display.max_columns', None)

df1 = download_data('0000934')
df2 = download_data('0000006')
df1 = reverse_data(df1)
df2 = reverse_data(df2)
df1.set_index(['date'], inplace=True)
df2.set_index(['date'], inplace=True)

# print(df1.info())
# print(df2.info())
plt.figure(figsize=(8,6), dpi=100)
plt.plot(df1['close'], label='Finance')
plt.plot(df2['close'], label='Real_estate')
plt.title('Trend chart', fontsize=12)
plt.xlabel('Time', fontsize=10)
plt.ylabel('Price', fontsize=10)
plt.legend()
plt.show()