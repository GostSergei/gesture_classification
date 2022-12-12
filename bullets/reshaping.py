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
from modules.bullets import load_data, ask_confirmation, decorator_script_wrap, update_3D_data





def do_reshaping__4_0(tensor_3D, n_coord, n_metric=None):  # # [X Y]
    tensor = tensor_3D
    tensor = tensor.reshape(tensor.shape[:2] + (-1, n_coord))
    return tensor


def do_reshaping__4_1(tensor_3D, n_coord, n_metric):   # [X Y Vx Vy]
    tensor = tensor_3D
    tensor = tensor.reshape(tensor.shape[:2] + (-1, n_coord))
    tensor = tensor.transpose([0, 1, 3, 2])
    tensor = tensor.reshape(tensor.shape[:3] + (-1,tensor.shape[-1]//n_metric,))
    tensor = tensor.transpose([0, 1, 4, 3, 2])  # [X Y Vx Vy]
    # tensor = tensor.transpose([0, 1, 4, 2, 3])  # [X Vx Y Vy] 

    tensor = tensor.reshape(tensor.shape[:3] + (-1,))
    return tensor

def do_reshaping__4_2(tensor_3D, n_coord, n_metric):  # [X Vx Y Vy]
    tensor = tensor_3D
    tensor = tensor.reshape(tensor.shape[:2] + (-1, n_coord))
    tensor = tensor.transpose([0, 1, 3, 2])
    tensor = tensor.reshape(tensor.shape[:3] + (-1,tensor.shape[-1]//n_metric,))
    # tensor = tensor.transpose([0, 1, 4, 3, 2])  # [X Y Vx Vy]
    tensor = tensor.transpose([0, 1, 4, 2, 3])  # [X Vx Y Vy] 

    tensor = tensor.reshape(tensor.shape[:3] + (-1,))
    return tensor

def do_reshaping__4(tensor_3D, n_coord, n_metric):
    if n_metric > 1:
        tensor = do_reshaping__4_2(tensor_3D, n_coord, n_metric)
    else:
        tensor = do_reshaping__4_0(tensor_3D, n_coord, n_metric)
    return tensor

def do_reshaping__5(tensor_3D, n_coord, n_metric):  # [X Vx Y Vy] 
    tensor = tensor_3D
    tensor = tensor.reshape(tensor.shape[:2] + (-1, n_coord))
    tensor = tensor.transpose([0, 1, 3, 2])
    tensor = tensor.reshape(tensor.shape[:3] + (-1,tensor.shape[-1]//2,))
    tensor = tensor.transpose([0, 1, 4, 3, 2])
    return tensor



   
@decorator_script_wrap
def reshape_to_nD(src, dst, n_coord, shape_type, n_metric=1):
    data = load_data(src)

    if shape_type.lower() == "4d":
        reshape_fun = do_reshaping__4
        fun_delta_shape = lambda data_tensor: [data_tensor.shape[-1]/(n_coord*n_metric),n_coord*n_metric]
    elif shape_type.lower() == "5d":
        reshape_fun = do_reshaping__5
        fun_delta_shape = lambda data_tensor: [data_tensor.shape[-1]/(n_metric*n_coord),n_metric, n_coord] 
        if n_metric == 1:
            print(f"Warning! for shape_type = 5D({shape_type}) and n_metric = 1, reshaping is a formal action!")  
    else:
        assert False, f"Error, bad shape_type: {shape_type}"
        
    for key in data.keys():
        if key[0].lower() == 'x':
            data_tensor = data[key]
            print(f"For {key}: {data_tensor.shape}, n_coord={n_coord}, n_metric={n_metric}:  {data_tensor.shape[-1]}->{fun_delta_shape(data_tensor)}")
            data_tensor = reshape_fun(data_tensor, n_coord, n_metric)
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
    
    parser.add_argument('-c', '--n_coord', default='3')
    parser.add_argument('-t', '--shape_type', default='4D',)
    parser.add_argument('-m', '--n_metric', default='1')
    
    
    arg = parser.parse_args()
    script_name = pathlib.Path(__file__).name
    
    
    good_shape_types = [ '4d', '5d']
    assert arg.shape_type.lower() in good_shape_types, f"Error! shape_type:{arg.shape_type} should be in {good_shape_types}"
    
    
    print(f"Start {script_name}:")
    # script function
    reshape_to_nD(src=arg.src, dst=arg.dst, n_coord=int(arg.n_coord), shape_type=arg.shape_type,
                  n_metric=int(arg.n_metric))
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()
    
    