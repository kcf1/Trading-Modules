import pandas as pd
import numpy as np

def split(label:pd.Series,features:pd.DataFrame,frac_train:float=0.8):
  df = pd.concat([label,features],axis=1).dropna()
  #train_end = df.index[round(df.shape[0]*frac_train)]
  
  train = df
  #train = df.loc[:train_end]
  #val = df.loc[train_end:]
  
  y_train,x_train = train.iloc[:,0],train.iloc[:,1:]
  #y_val,x_val = val.iloc[:,0],val.iloc[:,1:]
  
  return y_train,x_train#,y_val,x_val
  

from sklearn.ensemble import RandomForestClassifier

'''
def rf_select_fit(y_train:pd.Series,x_train:pd.DataFrame,
                  y_val:pd.Series,x_val:pd.DataFrame,
                  n_estimators:int=1000,max_depth:int=5,
                  frac:float=0.2):
  print('First fit...')
  rf = rf_fit(y_train,x_train,n_estimators,max_depth)
  print('Select features...')
  top_features = select_features(y_val,x_val,rf,frac)
  print('Refit...')
  rf_select = rf_fit(y_train,x_train.loc[:,top_features],n_estimators,max_depth)
  
  return rf_select
'''

def rf_fit(y_train:pd.Series,x_train:pd.DataFrame,
           n_estimators:int=1000,max_depth:int=5):
  rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
  rf.fit(y=y_train,
         X=x_train)
  
  return rf

from sklearn.inspection import permutation_importance

def select_features(y_val:pd.Series,x_val:pd.DataFrame,
                    rf,frac=0.2):
  feature_names = x_val.columns
  result = permutation_importance(rf,x_val,y_val,
                                  n_repeats=10,n_jobs=2)

  forest_importances = pd.Series(result.importances_mean,index=feature_names)
  top_n = round(len(feature_names) * frac)
  top_features = forest_importances.sort_values(ascending=False).index[:top_n]
  
  return top_features