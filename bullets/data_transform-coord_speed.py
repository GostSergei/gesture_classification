# Origin decomposition is a formal part for transfrom origin data in the common format  
# -s or --srs - a source of the data, file path
# -d or --dst - output file_name (extension should be .pkl as pickle lib was used to save the output data), file name 



import argparse
import os
import sys
import copy
from tabnanny import verbose
import numpy as np
import pathlib
import pickle
import time
import datetime

# you need to install your abs path the the parent folder of modules
sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
from modules.tensor_module import get_tucker_tensors
from modules.classification_auxiliary import update_df_to_coord_and_speed
from modules.bullets import load_data, ask_confirmation, decorator_script_wrap, update_3D_data


   
@decorator_script_wrap
def transform__coord_speed(src, dst, update_df_func, kwargs, verbose):
    data = load_data(src)
        
    for key in data.keys():
        if key[0].lower() == 'x':
            data_tensor = data[key]
            print("Old shape:",  data_tensor.shape)
             
            data_tensor = update_3D_data(data_tensor, update_df_func, kwargs=kwargs, verbose=verbose)

            print("New shape:",  data_tensor.shape)
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
    
    parser.add_argument('-v', '--verbose', default='1')
    # parser.add_argument('-c', '--n_coord', default='3')
    
    arg = parser.parse_args()
    script_name = pathlib.Path(__file__).name
 
    
    update_df_func = update_df_to_coord_and_speed
    kwargs={}
    
    
    print(f"Start {script_name}:")
    # script function
    transform__coord_speed(src=arg.src, dst=arg.dst, update_df_func=update_df_func, kwargs=kwargs,
                           verbose=int(arg.verbose))
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()
    
    