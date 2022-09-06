import argparse
import os
import sys
import time
import joblib as jbl
import os.path as osp
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
np.set_printoptions(threshold=sys.maxsize)


def load_data(src):
    data = np.load(src)
    return {
        'x_train': np.nan_to_num(data['x_train'], nan=0, posinf=0),
        'x_test': np.nan_to_num(data['x_test'], nan=0, posinf=0),
        'y_train': np.nan_to_num(data['y_train'], nan=0, posinf=0),
        'y_test': np.nan_to_num(data['y_test'], nan=0, posinf=0),
    }


def get_labels():
    with open('labels.txt', 'r') as inp:
        out = [int(line) for line in inp.readlines()]
    return list(set(out))


def get_classes_dict(gesutres_file_path='gestures.txt'):
    with open(gesutres_file_path, 'r') as inp:
        out = [line.strip() for line in inp.readlines()]
    # print(out)
    classes_dict = {}
    for i, class_ in enumerate(out):
        classes_dict[i] = class_        

    return classes_dict


def regress(data, dst=None):
    # data = load_data(src)
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    
    # model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=200).fit(x_train, y_train)
    x_test = StandardScaler().fit_transform(x_test)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    # model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000).fit(x_train, y_train)
    # y_pred = model.predict(x_train)
    y_pred = model.predict(x_test)
    
    if dst is not None:
        dst_folder = Path(dst).parent.absolute()
        os.makedirs(dst_folder, exist_ok=True)
        jbl.dump(model, dst)
        
    return model, y_pred
    
    
def randforest(data, dst=None, seed=None):
    # data = load_data(src)
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    
    # model = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=200).fit(x_train, y_train)
    x_test = StandardScaler().fit_transform(x_test)
    x_train = StandardScaler().fit_transform(x_train)
    model = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=seed).fit(x_train, y_train)
    y_pred = model.predict(x_train)
    
    if dst is not None:
        dst_folder = Path(dst).parent.absolute()
        os.makedirs(dst_folder, exist_ok=True)
        jbl.dump(model, dst)
        
    return model, y_pred


def evaluate(data, model, txt_file=None):
    # data = load_data(data_file)
    x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = StandardScaler().fit_transform(x_test)
    # x_train = StandardScaler().fit_transform(x_train)
    # model = jbl.load(model_file)
    start_time = time.time()
    y_pred = model.predict(x_test)
    stop_time = time.time() - start_time
    print('inference time:', stop_time)
    return y_pred
    # confmat = confusion_matrix(y_test, y_pred)
    # # pd.DataFrame(confmat).to_csv('output/confmat_stratified.txt', index=False)
    # # dst_folder = Path(txt_file).parent.absolute()
    # # os.makedirs(dst_folder, exist_ok=True)
    # # with open(txt_file, 'w') as out:
    # print('Accuracy:', accuracy_score(y_test, y_pred), file=out)
    # print('Time:', stop_time, file=out)
    # print('Samples:', len(y_test), file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default='input/data.npz')
    parser.add_argument('-d', '--dst', default='output/regress.jbl')
    arg = parser.parse_args()
    regress(arg.src, arg.dst)
    '''
    evaluate('input/mpipe-stratified.npz', 'output/regress-mpipe-stratified.jbl', 'output/evaluate-mpipe-stratified.txt')
    '''
