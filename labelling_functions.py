

import pandas as pd
import numpy as np
#Initiates a class that is capable of amending a dataframe in place. Returns only strings that are column names. 

class dgp():
    def __init__(self, df, price,volume):
        self.df = df
        self.price = df[price]
        self.volume = df[volume]
    
    def calc_consecutive_vols(self):
        string = "consecutive_vol_increases"
        self.df[string] = 0
        self.df.at[(self.volume>self.volume.shift(1)),string] = 1
        self.df.at[(self.df[string]==1)&
                           (self.volume.shift(1)>self.volume.shift(2)),string] = 2
        self.df.at[(self.df[string]==2)&
                           (self.volume.shift(2)>self.volume.shift(3)),string] = 3
        self.df.at[(self.df[string]==3)&
                           (self.volume.shift(3)>self.volume.shift(4)),string] = 4
        self.df.at[(self.df[string]==4)&
                           (self.volume.shift(4)>self.volume.shift(5)),string] = 5
        return string
        
    def calc_is_rolling_max(self,window):
        string = "is_rolling_max_"+str(window)
        self.df[string] = self.price.rolling(window).max() == self.price
        self.df[string] = self.df[string].apply(lambda x: 1 if x==True else 0)
        return string
    
    
    
    def calc_vol_vs_lta(self,window):
        string_ = "vol_vs_lta_last_"+str(window)
        self.df[string_] = (self.volume/self.volume.rolling(window).mean())-1
        return string_
    
    def calc_last_avg_vol(self,window):
        string = "avg_vol_last_"+str(window)
        self.df[string]=self.volume.rolling(window).mean()
        return string
    
    def calc_price_change(self,df,price,window):
        string_ = "price_change_last_"+str(window)
        self.df[string_] = (self.price/self.price.shift(window))-1
        return  string_
    
    def fwd_max_pct_label(self,window,threshold):
        string = "max_price_change_"+str(window)+"_"+str(threshold)
        self.df[string] = (self.price.rolling(window).max().shift(-window)/self.price)-1
        self.df[string] = self.df[string].apply(lambda x: 1 if x>threshold else 0)
        return string
    
    def trailing_stop_loss_tp(self,sl,windows,label_threshold): #calculates binary label based on sl and trailing sl
        string = "sl_label_"+str(sl)+"_"+str(windows)+"_"+str(label_threshold)
        self.df[string] = None
        self.df[string+"_pct_return"] = None
        output = []
        for i in range(0,len(self.df)-windows):
            #print(i)
            prices = self.price[i:i+windows]
            #print(len(prices))
            prices.reset_index(inplace=True,drop=True)
            start_price = prices[0]
            net_changes = (prices/prices[0])-1
            cummax = prices.cummax()
            low_changes = pd.DataFrame(np.array(prices)*(1/np.array(cummax)))-1
            out_minute = low_changes[low_changes[0] <=-sl].first_valid_index()
            if out_minute == None:
                pct_return = (prices[windows-1]/start_price)-1
                #out_minute = windows
            else:
                pct_return = (prices[out_minute]/start_price)-1
                #out_minute +=i
            self.df.loc[i,string] = (pct_return > label_threshold)
            self.df.loc[i,string+"_pct_return"] = pct_return
        self.df[string] = self.df[string].apply(lambda x: 1 if x ==True else 0)
        return string,string+"_pct_return"


# In[ ]:


random_data = pd.DataFrame(np.array([np.random.random_sample((100)),np.random.random_sample((100))]).T,columns =['price','volume'])
new_dgp = dgp(random_data,'price','volume')
new_dgp.calc_consecutive_vols()

