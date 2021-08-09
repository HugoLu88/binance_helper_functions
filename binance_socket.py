#!/usr/bin/env python
# coding: utf-8

# In[1]:


from binance.client import Client


class binancePublicSocket(Client):
    #client for the binance websocket. Writes subscribed data to shared array mem in threadsafe manner
    def __init__( self,storage,storage_lock,api_key=None,api_secret=None):
        super().__init__( api_key, api_secret, requests_params=None, tld='com')
        self.storage = storage
        self.storage_lock = storage_lock
        
    def on_open(self):
        self.url = []
        self.message_count = 0
    # Below to be amended for binance
    def on_message(self,msg):
        self.message_count +=1
        
        if msg['type'] == "ticker":
            trade_details = { key : msg[key] for key in required_keys } # extract required info from ticker message
            self.storage_lock.acquire()
            self.storage.append(trade_details)
            self.storage_lock.release()
    
    def on_close(self):
        print("-- Goodbye! --")
        

        

