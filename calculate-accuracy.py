import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib import style
import datetime as dt
import pandas_datareader.data as web
import glob, os
from pandas.plotting import register_matplotlib_converters
import wget
from contextlib import closing
from os.path import exists
from urllib import request
import statistics
import time

import bs4 as bs
import pickle
import requests

style.use('bmh')
register_matplotlib_converters()

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def date_loop(start, end):
    delta = dt.timedelta(days=1)
    date_list = []
    while start <= end:
        weekno = start.weekday()
        date_str = str(start).replace('-','')
        if weekno < 5:
            date_list.append(date_str)
        else:  # 5 Sat, 6 Sun
            pass
        start += delta

    return date_list

def grab_npsx_data(ticker, date_list, download=False):
    if download:
        for date in date_list:
            file = "NPSXshvol" + date + ".txt"
            if not exists("nasdaq-psx/"+file):
                url = 'ftp://ftp.nasdaqtrader.com/files/shortsaledata/daily/psx/' + file
                try:
                    wget.download(url,out="nasdaq-psx")
                except:
                    print("Date "+date+" does not exist")
                    continue

    d = []
    for file in sorted(glob.glob("nasdaq-psx/*.txt")):
        df = pd.read_csv(file, sep="|", engine="python")
        df.drop(['MARKET'], axis=1, inplace=True)
        df.drop(['TOTAL VOLUME'], axis=1, inplace=True)
        df.set_index("SYMBOL", inplace=True)
        df['DATE'] =  pd.to_datetime(df['DATE'], format = '%Y%m%d')

        d.append({'Date': df.loc[ticker,'DATE'], 'N_PSX_SV': df.loc[ticker,'SHORT VOLUME']})

    ticker_data_frame = pd.DataFrame(d)
    ticker_data_frame.iloc[:,1] *= 100 #Multiplier for scaling issues
    ticker_data_frame.set_index("Date", inplace=True)

    return ticker_data_frame

def grab_nbx_data(ticker, date_list, download=False):
    if download:
        for date in date_list:
            file = "NQBXshvol" + date + ".txt"
            if not exists("nasdaq-nbx/"+file):
                url = 'ftp://ftp.nasdaqtrader.com/files/shortsaledata/daily/bx/' + file
                try:
                    wget.download(url,out="nasdaq-nbx")
                except:
                    print("Date "+date+" does not exist")
                    pass

    d = []
    for file in sorted(glob.glob("nasdaq-nbx/*.txt")):
        df = pd.read_csv(file, sep="|", engine="python")
        df.drop(['MARKET'], axis=1, inplace=True)
        df.drop(['TOTAL VOLUME'], axis=1, inplace=True)
        df.set_index("SYMBOL", inplace=True)
        df['DATE'] =  pd.to_datetime(df['DATE'], format = '%Y%m%d')

        d.append({'Date': df.loc[ticker,'DATE'], 'N_QBX_SV': df.loc[ticker,'SHORT VOLUME']})

    ticker_data_frame = pd.DataFrame(d)
    ticker_data_frame.iloc[:,1]  *= 100 #Multiplier for scaling issues
    ticker_data_frame.set_index("Date", inplace=True)

    return ticker_data_frame

def grab_finra_data(ticker, date_list,download=False):
    if download:
        opener = request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        request.install_opener(opener)
        for date in date_list:
            file = "CNMSshvol" + date + ".txt"
            if not exists("finra/"+file):
                dir = "finra/"+file
                url = 'http://regsho.finra.org/' + file
                try:
                    request.urlretrieve(url, dir)
                except:
                    print("Date "+date+" does not exist")
                    pass

    d = []
    for file in sorted(glob.glob("finra/*.txt")):
        df = pd.read_csv(file, sep="|", skipfooter=1, engine="python")
        df.drop(['Market'], axis=1, inplace=True)
        df.drop(['ShortExemptVolume'], axis=1, inplace=True)
        df.set_index("Symbol", inplace=True)
        df['Date'] =  pd.to_datetime(df['Date'], format = '%Y%m%d')

        d.append({'Date': df.loc[ticker,'Date'], 'Finra_SV': df.loc[ticker,'ShortVolume']})

    ticker_data_frame = pd.DataFrame(d)
    ticker_data_frame.set_index("Date", inplace=True)

    return ticker_data_frame


def grab_stock_data(ticker, d_list, download = False):
    start = d_list[0]
    end = d_list[-1]
    sp_df = web.DataReader(ticker, 'yahoo', start, end)
    sp_df.to_csv('ticker-data/'+ticker+'.csv')

    sp_df = pd.read_csv('ticker-data/'+ticker+'.csv', parse_dates=True,index_col=0)
    sp_df['pct_chng'] = (sp_df['Close'] - sp_df['Open']) / sp_df['Open']
    sp_df = sp_df[["pct_chng"]]

    return sp_df

#Handle for holidays!
def algorithm_accuracy(start, end, data_df, threshold = 0):
    delta = dt.timedelta(days=1)
    delta_two = dt.timedelta(days=2)
    current_date = start + delta
    past_date = start
    future_date = current_date + delta
    success = 0
    total = 0

    while current_date <= end:
        current_during_week = current_date.weekday()
        if current_during_week < 5 and current_during_week > 0:
            date_range_df = data_df.loc[past_date : current_date]
            corr_df = date_range_df.drop(['pct_chng'], axis=1).corr()

            if corr_df.loc['Finra_SV', 'N_PSX_SV'] == 1.0 and corr_df.loc['Finra_SV', 'N_QBX_SV'] == 1.0:
                try:
                    psx_change = ( data_df.loc[str(current_date), 'N_PSX_SV'] - data_df.loc[str(past_date), 'N_PSX_SV'] ) / data_df.loc[str(past_date), 'N_PSX_SV']
                    finra_change = ( data_df.loc[str(current_date), 'Finra_SV'] - data_df.loc[str(past_date), 'Finra_SV'] ) / data_df.loc[str(past_date), 'Finra_SV']
                    qbx_change = ( data_df.loc[str(current_date), 'N_QBX_SV'] - data_df.loc[str(past_date), 'N_QBX_SV'] ) / data_df.loc[str(past_date), 'N_QBX_SV']
                    changes = [psx_change, finra_change, qbx_change]
                    avg_change = statistics.mean(changes)
                    #print("Average change:", avg_change)

                    #Requirement passed: Correlation with positive movement
                    if avg_change > 0:
                        changes_var = np.var(changes)

                        #Now check if the next day return is negative - in other words, check if the theory is correct for this specific day
                        if (data_df.loc[str(future_date), 'pct_chng']) < threshold:
                            success+=1

                        total+=1
                except:
                    pass

        past_date = current_date
        current_date += delta
        future_date += delta

    accuracy = success/total

    return accuracy


def ticker_average(start, end, threshold, ticker_list, d_list, download):
    accuracy = []

    l = len(ticker_list)

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for i, ticker in enumerate(ticker_list):
        n_psx_df = grab_npsx_data(ticker, d_list, download)
        n_bx_df = grab_nbx_data(ticker, d_list, download)
        finra_df = grab_finra_data(ticker, d_list, download)
        price_df = grab_stock_data(ticker, d_list, download)
        all_data_df = finra_df.join([n_psx_df, n_bx_df, price_df])

        accuracy.append(algorithm_accuracy(start, end, all_data_df, threshold))

        # Update Progress Bar
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    average_accuracy = statistics.mean(accuracy)
    print("Accuracy: " + str(average_accuracy*100) + "%")
    return average_accuracy

def graph_data(ticker, d_list, download):
    n_psx_df = grab_npsx_data(ticker,d_list, download)
    n_bx_df = grab_nbx_data(ticker,d_list, download)
    finra_df = grab_finra_data(ticker, d_list, download)
    price_df = grab_stock_data(ticker, d_list, download)
    all_data_df = finra_df.join([n_psx_df, n_bx_df, price_df])

    plt.figure(1, figsize=(20,10))
    barchart = plt.bar(all_data_df.index, all_data_df['pct_chng'], color='blue')
    plt.ylabel("Price Percent Change")
    plt.xlabel("Date")
    plt.twinx()
    plt.ylabel("Short Volume")
    linechart = plt.plot(all_data_df.index, all_data_df['Finra_SV'], color='red', label="Finra")
    linechart2 = plt.plot(all_data_df.index, all_data_df['N_PSX_SV'], color='green', label="PSX")
    linechart3 = plt.plot(all_data_df.index, all_data_df['N_QBX_SV'], color='black', label="BX")

    #Chart Details
    plt.legend()

    plt.show()


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.find_all('td')[0].text.replace('\n','')
        tickers.append(ticker)

    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    return tickers


#Options
start = dt.date(2020, 12, 1)
end = dt.date(2021, 2, 12)
download = False #Change to true if you set the date further back
graph = False #Change to true to use Graph
threshold = 0.00 #Dip threshold (percent decrease in price to check)
ticker_list = ["APHA", "KT", "SNDL", "TLRY", "PLTR", "OGI", "MVIS", "NIO", "SENS", "TSLA", "SPCE", "DKNG", "ON", "AMAT", "AMD", "OCGN", "MARA"]
#ticker_list = save_sp500_tickers() #Uncomment if using the entire SP500 list
#Note that some tickers in the SP500 don't trade on the NASDAQ, so you may get a "Ticker not found" error

d_list = date_loop(start, end)
average_accuracy = ticker_average(start, end, threshold, ticker_list, d_list, download)

graph_ticker = "TSLA" #Ticker to graph
if graph:
    graph_data(ticker, d_list, download)
