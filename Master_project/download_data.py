import pandas as pd
import numpy as np
import urllib.request, datetime

today = datetime.date.today()  # Set 'today' to the current date


def get_page(url):  # Define a function that takes a url argument
    req = urllib.request.Request(url, headers={
        'Connection': 'Keep-Alive',
        'Accept': 'text.py/html, application/xhtml+xml, */*',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    })  # Create a request object with the specified headers
    opener = urllib.request.urlopen(req)  # Use the request object to open the url
    page = opener.read()  # Read the content of the page
    return page


def download_data(index: str):  # Define a function that downloads the data of 'index'
    url = 'http://quotes.money.163.com/service/chddata.html?code=%s&start=20090101&end=%s&fields=' \
          'TCLOSE;HIGH;LOW;LCLOSE;CHG;PCHG;VOTURNOVER;VATURNOVER' % (index, today.strftime("%Y%m%d"))

    raw_data = get_page(url)  # Call the 'get_page' function with the url and save the content
    raw_data = str(raw_data, 'utf-8')
    file = open('raw_data.txt', 'w')
    file.write(raw_data)
    file.close()

    df = pd.read_csv('raw_data.txt')  # Read the 'raw_data.txt' file into a pandas dataframe
    df.drop(columns=['股票代码', '名称'], axis=1, inplace=True)
    df.columns = ['date', 'close', 'high', 'low', 'pre_close', 'range', 'change', 'vol', 'turnover']
    # Rename the remaining columns of the dataframe
    df = df.replace(to_replace='None', value=np.nan)  # Replace any 'None' values with 'NaN'
    df.dropna(axis=0, how='any', inplace=True)  # Drop any rows that contain missing values
    df[['pre_close', 'range', 'change']] = df[['pre_close', 'range', 'change']].astype(float)
    # Convert the specified columns of the dataframe to a float datatype
    df.to_csv('raw_data.csv', index=False)
    return df
