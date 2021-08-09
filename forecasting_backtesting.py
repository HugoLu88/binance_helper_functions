#!/usr/bin/env python
# coding: utf-8

# In[2]:


from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from scipy.stats import kstest
import statsmodels.api as sm
import helper_functions
import statistics
import numpy as np
import pandas as pd

class forecasting():
    #client for the binance websocket. Writes subscribed data to shared array mem in threadsafe manner
    def __init__( self):
        print("initiating forecasting")
        
        
    def calibrate_ou(self,Y):
        my_stats = statistics.statistics()
        c,a = my_stats.autoregression(Y,1,'c').params[0],my_stats.autoregression(Y,1,'c').params[1]
        yhat = c + a*Y.shift(1)
        residuals = (Y-yhat).dropna()
        b = residuals.std()
        chg_t = 1
        
        theta = (1-a)/chg_t
        mu = c/(theta*chg_t)
        
        sigma = b/(chg_t**0.5)
        return theta,mu,sigma
    
    def sim_ou(self,start,theta,mu,sigma,sims,length):
        increments = np.random.normal(0,sigma,[length,sims])
        output = np.zeros([length,sims])
        output[0] = start
        thetamu = np.ones([length,sims])*theta*mu
        
        for i in range(1,length):
            output[i] = output[i-1]+thetamu[i] - output[i-1]*theta + increments[i]
        return output.T
    
    def sim_ar1(self,start,beta,sigma,sims,length,constant = None):
        if constant is None:
            constant = 0  
        increments = np.random.normal(0,sigma,[length,sims])
        output = np.zeros([length,sims])
        output[0] = start


        for i in range(1,length):
            output[i] = output[i-1]*beta + increments[i] + constant
        return output.T


        
            
        
        
    
class backtesting():
    #client for the binance websocket. Writes subscribed data to shared array mem in threadsafe manner
    def __init__( self):
        print("initiating backtesting")
     
    def start_arb_trading(self,X1,X2,signal,upper,lower,stop,capital,signal_roll):
        #Placeholders#
        longX1prices = []
        longX2prices = []
        shortX1prices = []
        shortX2prices = []
      
        #Trade enter parameters#
        entrytime,entrypriceX1,entrypriceX2 = [],[],[]
        #Exit trade parameters#
        exittime,exitpriceX1,exitpriceX2=[],[],[]
        trade_type = []
        #Capital
       
        #Note the signal is X2 hat minus X2. hence, if the signal is positive, this implies X2 is to be shorted and X1 longed
        for i in range(signal_roll,len(signal)):
            s = signal[i]
            if np.absolute(s) <(stop): #The liquidation signal is the same for all trades, so liquidate
               
                for j in range(0,len(longX2prices)): #For each longX2 trade logged, log the exit price for X1,X2 and exit time
                    exitpriceX1.append(X1[i])
                    exitpriceX2.append(X2[i])
                    exittime.append(i)
                for j in range(0,len(longX1prices)): #For each longX1 trade logged, log the exit price for X1,X2 and exit time
                    exitpriceX1.append(X1[i])
                    exitpriceX2.append(X2[i])
                    exittime.append(i)
                #Wipe the placeholders
                longX1prices = []
                longX2prices = []
                shortX1prices = []
                shortX2prices = []
                
            x2p = X2[i]
            x1p = X1[i]
            x2size = capital/x2p
            x1size = capital/x1p
            if s <lower: #Implies X2 is high, so X2 to be shorted and X1 longed
                shortX2prices.append(x2p)
                longX1prices.append(x1p)
                entrytime.append(i)
                entrypriceX1.append(x1p)
                entrypriceX2.append(x2p)
                trade_type.append("Long X1")
            if s >upper : #Implies X2 is low, so X2 to be longed and X1 shorted
                shortX1prices.append(x1p)
                longX2prices.append(x2p)
                entrytime.append(i)
                entrypriceX1.append(x1p)
                entrypriceX2.append(x2p)
                trade_type.append("Long X2")
                
        out_ = [entrytime,entrypriceX1,entrypriceX2 ,exittime,exitpriceX1,exitpriceX2,trade_type]        
        out_ = pd.DataFrame(out_).transpose().dropna().reset_index(drop=True)
        out_ = out_.rename(columns = {
        0:"entrytime",
        1:"entrypriceX1",
        2:"entrypriceX2" ,
        3:"exittime",
        4:"exitpriceX1",
        5:"exitpriceX2",
        6:"trade_type"
        })
        out_['entrytime']=out_['entrytime'].astype('int64')
        floats = ["entrypriceX1","entrypriceX2" ,"exittime","exitpriceX1","exitpriceX2"]
        for i in floats:
            out_[i]=    out_[i].astype('float64')
            
        out_['x1position'] = capital/out_['entrypriceX1']
        out_['x2position'] = capital/out_['entrypriceX2']
        out_['longX1profit'] = (out_['exitpriceX1']-out_['entrypriceX1'])*out_['x1position']
        out_['shortX2profit'] = (out_['exitpriceX2']-out_['entrypriceX2'])*out_['x2position']*-1
        out_['ttl_lx1_profit']=out_['shortX2profit']+out_['longX1profit'] 

        out_['longX2profit'] = (out_['exitpriceX2']-out_['entrypriceX2'])*out_['x2position']
        out_['shortX1profit'] = (out_['exitpriceX1']-out_['entrypriceX1'])*out_['x1position']*-1
        out_['ttl_lx2_profit']=out_['shortX1profit']+out_['longX2profit'] 
        
        return out_
                
    def start_arb_trading_v2(self,X1,X2,signal,upper,lower,tp,capital,signal_roll):
        #Placeholders#
        longX1prices = []
        longX2prices = []
        shortX1prices = []
        shortX2prices = []
      
        #Trade enter parameters#
        entrytime,entrypriceX1,entrypriceX2 = [],[],[]
        #Exit trade parameters#
        exittime,exitpriceX1,exitpriceX2=[],[],[]
        trade_type = []
        #Capital
        returns  = []
        #Note the signal is X2 hat minus X2. hence, if the signal is positive, this implies X2 is to be shorted and X1 longed
        for i in range(signal_roll,len(signal)):
            s = signal[i]
            x2p = X2[i]
            x1p = X1[i]
            #Step 1 ------------------------ ------------------------  ------------------------  ------------------------  ----
            #First, generate the vector of returns for the first type of position
            x1longpct = x1p/(np.array(longX1prices))-1
            x2shortpct = x2p/(np.array(shortX2prices))-1
            total_x1_long_pct = x1longpct+x2shortpct
            #Then, generate the total percentage return on the position
            total_x1_long_returns = total_x1_long_pct[np.where(total_x1_long_pct>tp)[0]]
            for j,jtem in enumerate(total_x1_long_returns):
                returns.append(jtem)
                exitpriceX1.append(x1p)
                exitpriceX2.append(x2p)
                exittime.append(i)
            #Add each element where there is a return over the threshold to our returns. Delete those entries from the price lists. Do not forget to add the exitprices and the exit times
            longX1prices = list(np.delete(longX1prices,np.where(total_x1_long_returns>tp)[0]))
            shortX2prices = list(np.delete(shortX2prices,np.where(total_x1_long_returns>tp)[0]))
            #Step 2 ------------------------ ------------------------  ------------------------  ------------------------  ----
            x2longpct = x2p/(np.array(longX2prices))-1
            x1shortpct = x1p/(np.array(shortX1prices))-1
            total_x2_long_pct = x2longpct+x1shortpct
            #Then, generate the total percentage return on the position
            total_x2_long_returns = total_x2_long_pct[np.where(total_x2_long_pct>tp)[0]]
            for j,jtem in enumerate(total_x2_long_returns):
                returns.append(jtem)
                exitpriceX1.append(x1p)
                exitpriceX2.append(x2p)
                exittime.append(i)
            #Add each element where there is a return over the threshold to our returns. Delete those entries from the price lists. Do not forget to add the exitprices and the exit times
            longX2prices = list(np.delete(longX2prices,np.where(total_x2_long_returns>tp)[0]))
            shortX1prices = list(np.delete(shortX1prices,np.where(total_x2_long_returns>tp)[0]))



            x2size = capital/x2p
            x1size = capital/x1p
            
            ####This calculates the purchases / opening of positions
            if s <lower: #Implies X2 is high, so X2 to be shorted and X1 longed
                shortX2prices.append(x2p)
                longX1prices.append(x1p)
                entrytime.append(i)
                entrypriceX1.append(x1p)
                entrypriceX2.append(x2p)
                trade_type.append("Long X1")
            if s >upper : #Implies X2 is low, so X2 to be longed and X1 shorted
                shortX1prices.append(x1p)
                longX2prices.append(x2p)
                entrytime.append(i)
                entrypriceX1.append(x1p)
                entrypriceX2.append(x2p)
                trade_type.append("Long X2")
                
        out_ = [entrytime,entrypriceX1,entrypriceX2 ,exittime,exitpriceX1,exitpriceX2,trade_type]        
        out_ = pd.DataFrame(out_).transpose().dropna().reset_index(drop=True)
        out_ = out_.rename(columns = {
        0:"entrytime",
        1:"entrypriceX1",
        2:"entrypriceX2" ,
        3:"exittime",
        4:"exitpriceX1",
        5:"exitpriceX2",
        6:"trade_type"
        })
        out_['entrytime']=out_['entrytime'].astype('int64')
        floats = ["entrypriceX1","entrypriceX2" ,"exittime","exitpriceX1","exitpriceX2"]
        for i in floats:
            out_[i]=    out_[i].astype('float64')
            
        out_['x1position'] = capital/out_['entrypriceX1']
        out_['x2position'] = capital/out_['entrypriceX2']
        out_['longX1profit'] = (out_['exitpriceX1']-out_['entrypriceX1'])*out_['x1position']
        out_['shortX2profit'] = (out_['exitpriceX2']-out_['entrypriceX2'])*out_['x2position']*-1
        out_['ttl_lx1_profit']=out_['shortX2profit']+out_['longX1profit'] 

        out_['longX2profit'] = (out_['exitpriceX2']-out_['entrypriceX2'])*out_['x2position']
        out_['shortX1profit'] = (out_['exitpriceX1']-out_['entrypriceX1'])*out_['x1position']*-1
        out_['ttl_lx2_profit']=out_['shortX1profit']+out_['longX2profit'] 
        
        return out_
                