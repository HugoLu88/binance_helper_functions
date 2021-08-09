from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
from scipy.stats import kstest
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import helper_functions
from scipy.stats import ks_2samp

class statistics():
    #client for the binance websocket. Writes subscribed data to shared array mem in threadsafe manner
    def __init__( self):
        print("initiating stats")
        
    def autoregression(self,X,lags_,trend_):
        result = AutoReg(X,lags = lags_, trend = trend_).fit()
        return result
        
    def return_regression(self,X,Y):
        reg = LinearRegression().fit(X, Y)
        return reg
    
    def return_OLS(self,X,Y,add_constant):
        if add_constant == True:
            X = sm.add_constant(X)
        else:
            X = X
        model = sm.OLS(Y,X)
        results = model.fit()
        return results
    
    def adfuller(self,X,trend_):
        result = adfuller(X,regression=trend_)
        return result
    
    def check_cointegration(self,X,Y,trend_):
        #Note this assumes X and Y are I(1) i.e. random walks
        coin_result = coint(X, Y,trend=trend_) 
        return coin_result
        
        
    def full_cointegration(self,X,Y):
        new_helper = helper_functions.helper()
        trend_ = new_helper.input_bool("String, 'c' or 'ct': Use trend?")
        constant_ = new_helper.input_bool("Boolean: Use constant? Should be true")
        first_step_results = self.return_OLS(X,Y,constant_)
        if constant:
            beta = results.params[0]
        else:
            beta = results.params[1]
        difference = Y - beta*X
        
    def ks_2samp(self,X1,X2):
        result = ks_2samp(X1,X2)
        return result
