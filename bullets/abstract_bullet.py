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
# from modules.tensor_module import get_tucker_tensors
from modules.bullets import load_data, ask_confirmation, decorator_script_wrap


   
@decorator_script_wrap 
def script_fun(src, dst):
    pass



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
    script_fun(src=arg.src, dst=arg.dst)
    ###
    print(f"Finished script: {script_name}")
    print()


if __name__ == '__main__': 
    main()
    