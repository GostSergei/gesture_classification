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
from modules.bullets import load_data, ask_confirmation, decorator_script_wrap


   
@decorator_script_wrap
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
            print()
            pickle.dump(data_tucker, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'The output was saved: {dst}')
            # np.save(f, data_tucker, allow_pickle=True)

            
    return data_tucker



   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-d', '--dst', default=None)
    parser.add_argument('-r', '--rank',  default=-1,
                        help="For example rank='[120, 140]' or rank = '120' (equals to [120, 120])"\
                                ", default: -1(maximum rank)")
    arg = parser.parse_args()
    script_name = pathlib.Path(__file__).name
    
    print(f"Start {script_name}:")
    # script function
    tucker_decomposition(src=arg.src, dst=arg.dst, rank=arg.rank)
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()
    
    