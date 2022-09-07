import argparse
import os
import sys
import copy
import numpy as np
import pathlib
import pickle

sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
from modules.tensor_module import get_tucker_tensors


def load_data(src):
    data = np.load(src)
    return {
        'x_train': np.nan_to_num(data['x_train'], nan=0, posinf=0),
        'x_test': np.nan_to_num(data['x_test'], nan=0, posinf=0),
        'y_train': np.nan_to_num(data['y_train'], nan=0, posinf=0),
        'y_test': np.nan_to_num(data['y_test'], nan=0, posinf=0),
    }
    
    
    
def tucker_decomposition(src, dst, rank=-1):
    data = load_data(src)
    data_tensor, data_tensor_test = data['x_train'], data['x_test']
    tensor_tucker, tensor_tucker_test = get_tucker_tensors(data_tensor, data_tensor_test, rank=rank)
    data_tucker = copy.deepcopy(data)
    data_tucker['x_train'] = tensor_tucker
    data_tucker['x_test'] = tensor_tucker_test
    
    if dst is not None:
        with open(dst, 'wb') as f:
            pickle.dump(data_tucker, f, protocol=pickle.HIGHEST_PROTOCOL)
            # np.save(f, data_tucker, allow_pickle=True)
            
    return data_tucker


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
        
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-d', '--dst', default=None)
    
    parser.add_argument('-r', '--rank',  default=None,
                        help="For example rank='[1,2]' or rank = '1', default: -1")
    
    arg = parser.parse_args()
    
    dst = pathlib.Path(arg.dst)
    if pathlib.Path(arg.dst).exists():
        if not ask_confirmation(f'Reload {arg.dst}?'):
            print('Script was stopped!')
            exit(0)
        else:
            print(f'{arg.dst} will be reloaded')
            
    tucker_decomposition(arg.src, arg.dst, arg.rank)
    exit(0)