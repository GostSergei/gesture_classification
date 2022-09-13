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

from sklearn.preprocessing import StandardScaler
import json
import pickle
import pathlib
np.set_printoptions(threshold=sys.maxsize)




def ask_confirmation(question_text,):
    out = None
    while out is None:
        str_ = input(question_text + ' [y/n]: ')
        if str_.lower() in ['y', 'yes']:
            out = True
        elif str_.lower() in ['n', 'no']:
            out = False
        else:
            print(f"{str_} is not valid anwser!" + ' [y/n]: ')
    return out


def load_data(src):
    data_ = np.load(src, allow_pickle=True)
    nan = 0
    data = {}
    for key in ['x_train', 'x_test', 'y_train', 'y_test']:
        data[key] = data_[key]
        if np.isnan(data[key]).sum() > 0:
            data[key] = np.nan_to_num(data[key], nan=nan, posinf=nan)
            print(f'For {key} nan will be replaced by {nan}!')
    return data

    
def load_json(src):
    with open(src, 'r') as f:
        out = json.load(f)
    return out


def save_json(data, dst):
    with open(dst, 'w') as f:
        json.dump(data, f)
            
def save_pickle(data, dst):
    with open(dst, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)




def get_labels():
    with open('labels.txt', 'r') as inp:
        out = [int(line) for line in inp.readlines()]
    return list(set(out))


def get_classes_dict(gesutres_file_path=None):
    
    if gesutres_file_path is None:
        out = _get_gesture_txt_list_()
    else:
        with open(gesutres_file_path, 'r') as inp:
            out = [line.strip() for line in inp.readlines()]

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
    
    
def _get_gesture_txt_list_():
    classes_str = "G01_left_handyes G02_right_handyes G03_left_handno G04_right_handno G05_left_select G06_right_select G07_left_call "\
                  "G08_right_call G09_left_mute G10_right_mute G11_left_unmute G12_right_unmute G13_left_close G14_right_close G15_left_wave "\
                  "G16_right_wave G17_left_write G18_right_write G19_headyes G20_headno G21_left_roll G22_right_roll G23_left_yaw G24_right_yaw G25_left_save "\
                  "G26_right_save G27_left_export G28_right_export G29_left_pupil G30_right_pupil G31_left_swipeup G32_right_swipeup G33_left_swipedown G34_right_swipedown "\
                  "G35_left_swipeleft G36_right_swipeleft G37_left_swiperight G38_right_swiperight G39_left_high G40_right_high G41_moveforward G42_movebackward "\
                  "G43_moveup G44_movedown G45_moveleft G46_moveright G47_screenshot G48_central_zoomin G49_central_zoomout G50_left_zoomin G51_left_zoomout "\
                  "G52_right_zoomin G53_right_zoomout G54_delete"
    classes = classes_str.split()
    return classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default='input/data.npz')
    parser.add_argument('-d', '--dst', default='output/regress.jbl')
    arg = parser.parse_args()
    regress(arg.src, arg.dst)
    '''
    evaluate('input/mpipe-stratified.npz', 'output/regress-mpipe-stratified.jbl', 'output/evaluate-mpipe-stratified.txt')
    '''






# def _get_gesture_txt_list_():
#     classes_str = "G01_left_handyes G02_right_handyes G03_left_handno G04_right_handno G05_left_select G06_right_select G07_left_call "\
#                   "G08_right_call G09_left_mute G10_right_mute G11_left_unmute G12_right_unmute G13_left_close G14_right_close G15_left_wave "\
#                   "G16_right_wave G17_left_write G18_right_write G19_headyes G20_headno G21_left_roll G22_right_roll G23_left_yaw G24_right_yaw G25_left_save "\
#                   "G26_right_save G27_left_export G28_right_export G29_left_pupil G30_right_pupil G31_left_swipeup G32_right_swipeup G33_left_swipedown G34_right_swipedown "\
#                   "G35_left_swipeleft G36_right_swipeleft G37_left_swiperight G38_right_swiperight G39_left_high G40_right_high G41_moveforward G42_movebackward "\
#                   "G43_moveup G44_movedown G45_moveleft G46_moveright G47_screenshot G48_central_zoomin G49_central_zoomout G50_left_zoomin G51_left_zoomout "\
#                   "G52_right_zoomin G53_right_zoomout G54_delete"
#     classes = classes_str.split()
#     return classes


