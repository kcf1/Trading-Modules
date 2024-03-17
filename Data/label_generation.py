import pandas as pd
import numpy as np

def label(r):
  if max(r) <= 0:
    return -1
  else:
    return np.argmax(r)

def map_name(k,names):
  if k == -1:
    return 'flat'
  else:
    return names[k]

def best_signal(scores:pd.DataFrame):
  signals = scores.columns
  best_signal = scores.apply(label,axis=1)
  
  print(best_signal.map(lambda k: map_name(k,signals)).value_counts())
  
  return best_signal
