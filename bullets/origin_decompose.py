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
def origin_decomposition(src, dst, nan):
    data = load_data(src, nan=nan)

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
    parser.add_argument('-n', '--nan', default='ffill')

    arg = parser.parse_args()
    script_name = pathlib.Path(__file__).name
    
    print(f"Start {script_name}:")
    # script function
    origin_decomposition(src=arg.src, dst=arg.dst, nan=arg.nan)
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()
    
    