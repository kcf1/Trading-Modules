import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas_datareader as pdr
from statsmodels import api as sm

def bootstrap_optimal(pnls:pd.DataFrame,frac:float=0.1,n_samples:int=100):
  pnls = pnls.dropna()
  sample_size = round(pnls.shape[0] * frac)
  
  bootstrap = pnls.sample(n=n_samples*sample_size, replace=True, axis=0).reset_index(drop=True)
  
  mu = bootstrap.groupby(bootstrap.index//sample_size).mean()
  sig = bootstrap.groupby(bootstrap.index//sample_size).std()
  sr = mu / sig
  
  optimal = sr.apply(lambda r: np.argmax(r),axis=1)
  
  rank = optimal.value_counts().sort_values(ascending=False)
  rank.index = rank.index.map(lambda x: bootstrap.columns[x])
  return rank
  