# Tucker-based data transformation 
# -s or --srs - a source of the data, file path
# -d or --dst - output file_name (extension should be .pkl as pickle lib was used to save the output data), file name 
# -r or --rank - decomposition rank. 
#     For example rank='[120, 140]' or rank = '120' (equals to [120, 120]), default: -1(maximum rank)



import argparse
import os
import sys
import copy
import numpy as np
import pathlib
import pickle
import time
import datetime

# you need to install your abs path the the parent folder of modules
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
    # print(f'Tucker_decomposition of {src}')
    data = load_data(src)
    data_tensor, data_tensor_test = data['x_train'], data['x_test']
    t = time.time()
    tensor_tucker, tensor_tucker_test = get_tucker_tensors(data_tensor, data_tensor_test, rank=rank)
    t = time.time() - t
    print(f"Decompose time: {t:.3f} s")
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
                        help="For example rank='[120, 140]' or rank = '120' (equals to [120, 120])"\
                             ", default: -1(maximum rank)")
    
    arg = parser.parse_args()
    
    dst = pathlib.Path(arg.dst)
    # if pathlib.Path(arg.dst).exists():
    #     if not ask_confirmation(f'Reload {arg.dst}?'):
    #         print('Script was stopped!')
    #         exit(0)
    #     else:
    #         print(f'{arg.dst} will be reloaded')
            
    # print(arg.src)
    # t = time.time()
    script_name = pathlib.Path(__file__).name
    ftime_str = '%y.%m.%d-%H:%M:%S'
    date_time = datetime.datetime.now()
    print(f"Start {script_name} for {arg.src}")
    print(f"Start time: {date_time.strftime(ftime_str)}")
    tucker_decomposition(arg.src, arg.dst, arg.rank)
    date_time_end = datetime.datetime.now()
    print(f"End: {script_name} at {date_time_end.strftime(ftime_str)} [{(date_time_end-date_time).total_seconds():.3f} s]")
    print()
    # t = time.time() - t
    # print(f"Work time: {t:.3f} s")
    exit(0)