import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import os
import copy

from abc import ABC, abstractmethod
import time
import json
import timeit

from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


import xgboost as xgb
import pystmm



def create_model(init_model_dict):
    model_type_str = init_model_dict['model_type']
    
    model_type = None
    model_type = eval(model_type_str)
    
    print(model_type)
    PARAMS_KEY = ModelAbstract.PARAMS_KEY
    if PARAMS_KEY not in init_model_dict.keys():
        init_model_dict[PARAMS_KEY] = {}
        
    model = model_type(**init_model_dict[PARAMS_KEY])
    print(model, 'was created')
    return model

class ModelAbstract(ABC):
    NAME_KEY = 'name'
    DO_SCALER_KEY = 'do_scaler'
    N_TRIALS_KEY = 'n_trials'
    PARAMS_KEY = 'params'
    ALL_PARAMS_KEY = 'all_params'
    
    def __init__(self, src=None, **kwargs):
        self.main_dict = {self.NAME_KEY: None,
                          self.DO_SCALER_KEY: False,
                          self.N_TRIALS_KEY: 1,
                          self.PARAMS_KEY: {},}
        if src is not None:
            self.load_model_dict(src)
            
        for key in kwargs.keys():
            self.main_dict[self.PARAMS_KEY][key] = kwargs[key]
            
        self.model = self.init_model()  # with updating self.main_dict
        
        self.params = self.main_dict[self.PARAMS_KEY]
        self.all_params = self.model.get_params()
        
        self.do_scaler = self.main_dict[self.DO_SCALER_KEY]
        self.name = self.main_dict[self.NAME_KEY]
        self.n_trials = self.main_dict[self.N_TRIALS_KEY]
        self.scaler = None
        
        self.fit_time=None
        
    def __str__(self) -> str:
        return self.name + ": " + str(self.params)
      
             
    # init methods 
    @abstractmethod
    def init_model(self):
        pass
    
    @abstractmethod
    def gen_default_params(self):
        pass
    
    
    def reshape_X_tensor(self, X_tensor):
        if len(X_tensor.shape) > 2:
            X_tensor = X_tensor.reshape(X_tensor.shape[0], -1)  
        return X_tensor       
    
    def fit(self, X_train, y_train, verbose=1):
        
        X_train = self.reshape_X_tensor(X_train)
        
        if self.do_scaler:
            self.scaler = StandardScaler().fit(X_train)
            X_train = self.scaler.transform(X_train)
          
        if verbose > 0:  
            print(f"Model {self.name} is fitting...")
        t = time.time()
        self.model.fit(X_train, y_train)
        self.fit_time = (time.time() - t)
        if verbose > 0:
            print(f"{self.fit_time} s passed")
        return self
             
    def predict(self, X):
        X = self.reshape_X_tensor(X)
        
        if self.do_scaler:
            X = self.scaler.transform(X)
            
        pred = self.model.predict(X)
        return pred
           
    def eval_model(self, X_train, y_train, X_test, y_test, n_trials=None, metric_fun=accuracy_score, metric_name='acc', show_presentage=True):
        
        if n_trials is None:
            n_trials = self.n_trials
            
        X_train = self.reshape_X_tensor(X_train)
        X_test = self.reshape_X_tensor(X_test)
        
        if self.do_scaler:
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        acc_array = []
        fit_times = []
        print(f"Modle {self.name} is evaluating... n_trials={n_trials}")
        for i in tqdm(range(n_trials)): 
            
            self.fit(X_train, y_train, verbose=0)
            fit_times += [self.fit_time]    
            pred = self.model.predict(X_test)
            acc = metric_fun(y_test, pred)
            acc_array += [acc]
        acc_array = np.array(acc_array)
        mean = acc_array.mean()
        std = acc_array.std(ddof=1)
        
        show_str= f"{mean:.4f}({std:.4f})"
        if show_presentage:
            show_str= f"{100*mean:.2f}({100*std:.2f})%"

        print(f"Mean {metric_name}: {show_str}" )
        print(f"Mean fitting times:{np.array(fit_times).mean()}" )
        return acc_array
        
        
        
    def eval_inference_time(self, X, n_timeit=100):
        
        print(f"Inference time evaluation for mocel {self.name}... n_timeit={n_timeit}")
        X= self.reshape_X_tensor(X)
        if self.do_scaler:
            X = self.scaler.transform(X)
            
        # model = self
        
        total_time = timeit.timeit('self.model.predict(X)', number=n_timeit, globals=locals())
        inference_time = total_time/n_timeit
        print(f"Inference time for model {self.name}: {1000*inference_time:.3f} ms")
        return inference_time
            
  
    def load_model_dict(self, src):
        with open(src, 'r') as f:
            main_dict = json.load(f)
        for key in main_dict.keys():
            self.main_dict[key] = main_dict[key]
        return main_dict



    
     
class LogReg(ModelAbstract):
    def gen_default_params(self):
        def_params = dict(n_jobs=-1)
        return def_params
        
    def init_model(self):
        self.main_dict[self.NAME_KEY] = 'LogReg'
        self.main_dict[self.DO_SCALER_KEY] = True
        self.main_dict[self.N_TRIALS_KEY] = 3
                
        params = self.gen_default_params()
        params_new = self.main_dict[self.PARAMS_KEY]
        for key in params_new.keys():
            params[key] = params_new[key]
        
        return LogisticRegression(**params)

  
class RandForest(ModelAbstract):
    def gen_default_params(self):
        def_params = dict(n_jobs=-1)
        return def_params
        
    def init_model(self):
        self.main_dict[self.NAME_KEY] = 'RandForest'
        self.main_dict[self.DO_SCALER_KEY] = False
        self.main_dict[self.N_TRIALS_KEY] = 10
                
        params = self.gen_default_params()
        params_new = self.main_dict[self.PARAMS_KEY]
        for key in params_new.keys():
            params[key] = params_new[key]
        
        return RandomForestClassifier(**params)
    
    
class XGBoost(ModelAbstract):
    def gen_default_params(self):
        def_params = dict(n_jobs=-1)
        return def_params
        
    def init_model(self):
        self.main_dict[self.NAME_KEY] = 'XGBoost'
        self.main_dict[self.DO_SCALER_KEY] = False
        self.main_dict[self.N_TRIALS_KEY] = 10
                
        params = self.gen_default_params()
        params_new = self.main_dict[self.PARAMS_KEY]
        for key in params_new.keys():
            params[key] = params_new[key]
        
        return xgb.XGBClassifier(**params)
  
    
class SVMClassifier(ModelAbstract):
    def gen_default_params(self):
        def_params = dict(kernel='linear')
        return def_params
        
    def init_model(self):
        self.main_dict[self.NAME_KEY] = 'SVMClassifier'
        self.main_dict[self.DO_SCALER_KEY] = True
        self.main_dict[self.N_TRIALS_KEY] = 10
                
        params = self.gen_default_params()
        params_new = self.main_dict[self.PARAMS_KEY]
        for key in params_new.keys():
            params[key] = params_new[key]
        
        return SVC(**params)
    
class KNN(ModelAbstract):
    def gen_default_params(self):
        def_params = dict(n_jobs=8)
        return def_params
        
    def init_model(self):
        self.main_dict[self.NAME_KEY] = 'KNN'
        self.main_dict[self.DO_SCALER_KEY] = True
        self.main_dict[self.N_TRIALS_KEY] = 1
                
        params = self.gen_default_params()
        params_new = self.main_dict[self.PARAMS_KEY]
        for key in params_new.keys():
            params[key] = params_new[key]
        
        return KNeighborsClassifier(**params)
    
    
    
    
class STMM(ModelAbstract):
    def reshape_X_tensor(self, X_tensor):
        print('STM reshape')
        if len(X_tensor.shape) < 3:
            X_tensor = X_tensor.reshape(list(X_tensor.shape) + [-1]) 
        return X_tensor
    
    def gen_default_params(self):
        def_params = dict(maxIter=2, tolSTM=1e-2, tol=1e-2)
        return def_params
        
    def init_model(self):
        self.main_dict[self.NAME_KEY] = 'STMM'
        self.main_dict[self.DO_SCALER_KEY] = False
        self.main_dict[self.N_TRIALS_KEY] = 3
                
        params = self.gen_default_params()
        params_new = self.main_dict[self.PARAMS_KEY]
        for key in params_new.keys():
            params[key] = params_new[key]
        
        return pystmm.classifier.STMM(**params)
        
        
                