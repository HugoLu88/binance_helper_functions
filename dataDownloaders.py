#!/usr/bin/env python
# coding: utf-8

# In[35]:




from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import helper_functions
import configparser
import threading
from binance_socket import binancePublicSocket

class dataDownloaders():
    #client for the binance websocket. Writes subscribed data to shared array mem in threadsafe manner
    def __init__(self,type_,binance_client = None):
        self.type = type_
        self.binance_client = binance_client
        
        
    def binance_data_pull(self,ticker_list,granularity,path):
        if granularity == "m":
            helper = helper_functions.helper()
            days_back = helper.input_float("days back")
            end_string = datetime.now().strftime("%d %b, %Y")
            start_string = (datetime.now() - timedelta(days=  days_back)).strftime("%d %b, %Y")
            klines = []
            for i in ticker_list:
                print("getting data for: "+str(i))
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_1MINUTE, start_string,end_string)) # this needs to be changed
            data = []
            utc_time = lambda x: datetime.fromtimestamp(x/1000).strftime("%Y-%m-%d %H:%M:%S")
            float_list = ["Open","High","Low","Close","Volume","Quote asset volume","Number of trades","Taker buy base asset volume","Taker buy quote asset volume"]
            for i in klines:
                rider =pd.DataFrame(i,columns = ["open_time","Open","High","Low","Close","Volume","Close time","Quote asset volume","Number of trades","Taker buy base asset volume","Taker buy quote asset volume","ignore"])
                rider['open_time'] = rider['open_time'].apply(utc_time)
                rider['Close time'] = rider['Close time'].apply(utc_time)
                for j in float_list:
                    rider[j] = rider[j].astype("float")
                data.append(rider)
            output_paths = []
            for i in range(0,len(data)):
                string_ = path + "/"+"raw_data_"+str(ticker_list[i])+"_"+datetime.now().strftime("%d_%b_%Y")
                data[i].to_csv(string_)
                output_paths.append(string_)
            print("finished!")
                                    
            return output_paths
        
    def binance_data_pull_granularity(self,ticker_list,granularity,path):

        helper = helper_functions.helper()
        days_back = helper.input_float("days back")
        end_string = datetime.now().strftime("%d %b, %Y")
        start_string = (datetime.now() - timedelta(days=  days_back)).strftime("%d %b, %Y")
        klines = []
        for i in ticker_list:
            print("getting data for: "+str(i))
            
            if granularity == "m":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_1MINUTE, start_string,end_string)) # this needs to be changed
            elif granularity == "3m":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_3MINUTE, start_string,end_string)) # this needs to be changed
            elif granularity == "5m":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_5MINUTE, start_string,end_string)) # this needs to be changed
            elif granularity == "15m":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_15MINUTE, start_string,end_string)) # this needs to be changed
            elif granularity == "30m":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_30MINUTE, start_string,end_string)) # this needs to be changed
            elif granularity == "1h":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_1HOUR, start_string,end_string)) # this needs to be changed
            elif granularity == "2h":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_2HOUR, start_string,end_string)) # this needs to be changed
            elif granularity == "4h":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_4HOUR, start_string,end_string)) # this needs to be changed
            elif granularity == "6h":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_6HOUR, start_string,end_string)) # this needs to be changed
            elif granularity == "8h":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_8HOUR, start_string,end_string)) # this needs to be changed
            elif granularity == "12h":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_12HOUR, start_string,end_string)) # this needs to be changed
            elif granularity == "1d":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_1DAY, start_string,end_string)) # this needs to be changed
            elif granularity == "3d":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_3DAY, start_string,end_string)) # this needs to be changed
            elif granularity == "1w":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_1WEEK, start_string,end_string)) # this needs to be changed
            elif granularity == "1M":
                klines.append(self.binance_client.get_historical_klines(i, Client.KLINE_INTERVAL_1MONTH, start_string,end_string)) # this needs to be changed
            else:
                print("error")
                return
                
                
                
        data = []
        utc_time = lambda x: datetime.fromtimestamp(x/1000).strftime("%Y-%m-%d %H:%M:%S")
        float_list = ["Open","High","Low","Close","Volume","Quote asset volume","Number of trades","Taker buy base asset volume","Taker buy quote asset volume"]
        for i in klines:
            rider =pd.DataFrame(i,columns = ["open_time","Open","High","Low","Close","Volume","Close time","Quote asset volume","Number of trades","Taker buy base asset volume","Taker buy quote asset volume","ignore"])
            rider['open_time'] = rider['open_time'].apply(utc_time)
            rider['Close time'] = rider['Close time'].apply(utc_time)
            for j in float_list:
                rider[j] = rider[j].astype("float")
            data.append(rider)
        output_paths = []
        for i in range(0,len(data)):
            string_ = path + "/"+"raw_data_"+str(ticker_list[i])+"_"+datetime.now().strftime("%d_%b_%Y")+"_"+str(granularity)
            data[i].to_csv(string_)
            output_paths.append(string_)
        print("finished!")

        return output_paths




# In[ ]:





# In[36]:



"""
config = configparser.ConfigParser()
config.read(r"/Users/Hugolu88/Documents/repos/binance/config_global.ini")
storage = []
key = config['PARAMETERS']['key']
scrt = config['PARAMETERS']['scrt']
storage_lock = threading.Lock()
client = binancePublicSocket(storage,storage_lock,api_key = key,api_secret = scrt)
path_ = "/Users/Hugolu88/Documents/repos/binance_v2/Data"
download =dataDownloaders("BINANCE",client)
download.binance_data_pull(['BTCUSDT'],"m",path_)
"""


# In[ ]:





# In[ ]:





# In[ ]:




