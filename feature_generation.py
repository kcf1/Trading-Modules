import pandas as pd
import numpy as np

def gen_all(bars:pd.DataFrame):
  ta = gen_ta(bars)
  dist = gen_dist(bars)
  fund = gen_fund()
  
  features = pd.concat([ta,dist,fund],axis=1)
  features.dropna(inplace=True)
  
  check_stationarity(features)
  
  return features

def normalize(features:pd.DataFrame):
  trend = features.rolling(252).mean()
  std = features.rolling(252).std()
  norm = (features - trend) / std
  
  return norm.dropna()

from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
  #print ('Results of Dickey-Fuller Test:')
  dftest = adfuller(timeseries, autolag='AIC')
  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
  for key,value in dftest[4].items():
      dfoutput['Critical Value (%s)'%key] = value
  #print(np.around(dfoutput,6))
  return dfoutput['p-value']
    
def check_stationarity(features:pd.DataFrame):
  print('Checking for stationarity')
  for name in features.columns:
    #print(name)
    p = adf_test(features[name])
    if p > 0.05:
      print(f'\nWarning! {name} is not stationary ({p} > 0.05)')

from sklearn.decomposition import PCA

def pca_transform(variations:pd.DataFrame,n_components:int):
  variations = variations.dropna()
  pca = PCA(n_components=n_components)
  #explained = pd.Series(pca.explained_variance_ratio_)
  #print(explained)
  #pca_variations = pd.DataFrame(pca.transform(variations),index=variations.index)
  return pca.fit_transform(variations)[-1,0]

def rolling_pca(variations:pd.DataFrame,n_components:int,min_window:int):
  variations = variations.dropna()
  pca_feature = pd.Series()
  for i in range(min_window,variations.shape[0]):
    #start = variations.index[i-min_window]
    end = variations.index[i]
    data = variations.loc[:end]
    pca_feature.loc[end] = pca_transform(data,n_components)
  return pca_feature

from ta.momentum import roc
from ta.momentum import rsi
from ta.momentum import stoch
from ta.momentum import williams_r

from ta.volatility import average_true_range
from ta.volatility import bollinger_pband
from ta.volatility import donchian_channel_pband
from ta.volatility import ulcer_index

from ta.trend import adx
from ta.trend import cci
from ta.trend import macd_diff

def gen_ta(bars:pd.DataFrame):
  logbars = np.log(bars)
  o,h,l,c = logbars.Open,logbars.High,logbars.Low,logbars.Close
  ta = pd.DataFrame()
  
  for w in [5,10,20,60,120,240]:
    # momentum
    ind = 'roc' + '_' + str(w)
    ta[ind] = roc(c,w)
    
    ind = 'rsi' + '_' + str(w)
    ta[ind] = rsi(c,w)
    
    ind = 'stoch' + '_' + str(w)
    ta[ind] = stoch(h,l,c,w)
    
    ind = 'willr' + '_' + str(w)
    ta[ind] = williams_r(h,l,c,w)
    
    # volatility
    ind = 'atr' + '_' + str(w)
    ta[ind] = average_true_range(h,l,c,w)
    
    ind = 'bollp' + '_' + str(w)
    ta[ind] = bollinger_pband(c,w)
    
    ind = 'doncp' + '_' + str(w)
    ta[ind] = stoch(h,l,c,w)
    
    ind = 'ui' + '_' + str(w)
    ta[ind] = williams_r(h,l,c,w)
    
    ind = 'vol' + '_' + str(w)
    ta[ind] = c.diff().rolling(w).std().ffill()
    
    # trend
    ind = 'adx' + '_' + str(w)
    ta[ind] = adx(h,l,c,w)
    
    ind = 'cci' + '_' + str(w)
    ta[ind] = cci(h,l,c,w)
    
    ind = 'macd' + '_' + str(w)
    ta[ind] = macd_diff(c,w,w//2,w//3)
  
  return ta.dropna(axis=1,how='all').dropna()

def gen_dist(bars:pd.DataFrame):
  logbars = np.log(bars)
  o,h,l,c = logbars.Open,logbars.High,logbars.Low,logbars.Close
  dist = pd.DataFrame()
  
  for w in [5,10,20,60,120,240]:
    ind = 'std' + '_' + str(w)
    dist[ind] = c.rolling(w).std()
    
    ind = 'skew' + '_' + str(w)
    dist[ind] = c.rolling(w).skew()
    
    ind = 'kurt' + '_' + str(w)
    dist[ind] = c.rolling(w).kurt()
    
  return dist.dropna(axis=1,how='all').dropna()

from pandas_datareader.famafrench import FamaFrenchReader as FFR
from yfinance import download

def gen_fund():
  fund = pd.DataFrame()
  
  #block_print()
  vix = download(['^VIX']).Close
  usd = download(['DX-Y.NYB']).Close
  evz = download(['^EVZ']).Close
  ff5 = FFR('F-F_Research_Data_5_Factors_2x3_daily',start='1990-01-01').read()[0]
  #enable_print()
  fund['VIX'] = vix
  fund['EVZ'] = evz
  fund['USD'] = np.log(usd).diff()
  fund['MKT'] = ff5['Mkt-RF']
  fund['SMB'] = ff5.SMB
  fund['HML'] = ff5.HML
  fund['RMW'] = ff5.RMW
  fund['CMA'] = ff5.CMA
  fund['RF'] = ff5.RF
  
  return fund.dropna(axis=1,how='all').dropna()









import sys, os

# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enable_print():
    sys.stdout = sys.__stdout__