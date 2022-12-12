# Origin decomposition is a formal part for transfrom origin data in the common format  
# -s or --srs - a source of the data, file path
# -d or --dst - output file_name (extension should be .pkl as pickle lib was used to save the output data), file name 



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
from modules.bullets import load_data, ask_confirmation, decorator_script_wrap


   
@decorator_script_wrap
def reshape_to_4D(src, dst, n_coord):
    data = load_data(src)
    for key in data.keys():
        if key[0].lower() == 'x':
            data_tensor = data[key]
            print(f"For {key}: {data_tensor.shape}, n_coord={n_coord}, {data_tensor.shape[-1]}->[{data_tensor.shape[-1]/n_coord}, {n_coord}]")
            data_tensor = data_tensor.reshape(data_tensor.shape[:2] + (-1, n_coord))
            print("Final shape:",  data_tensor.shape)
            data[key] = data_tensor
            
    if dst is not None:
        with open(dst, 'wb') as f:
            print()
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'The output was saved: {dst}')
            
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-d', '--dst', default=None)
    parser.add_argument('-c', '--n_coord', default=3)
    # parser.add_argument('-n', '--nan', default='ffill')

    arg = parser.parse_args()
    script_name = pathlib.Path(__file__).name
    
    print(f"Start {script_name}:")
    # script function
    reshape_to_4D(src=arg.src, dst=arg.dst, n_coord=int(arg.n_coord))
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()
    
    