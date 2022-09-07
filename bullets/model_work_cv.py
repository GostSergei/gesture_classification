import argparse
import os
import sys
import copy
import numpy as np
import pathlib
import json 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
from modules.models_module import create_model
from modules.bullets import get_classes_dict, load_data, load_json, save_json, save_pickle, ask_confirmation

from modules.results_auxiliary import get_target_files
from modules.tensor_module import get_tucker_tensors, get_SVD_tensors
# import modules.models_module 

        
        
def work_with_model_cv(src, m_src, dst, transformation, classes):
    
    cv_scr_list = get_target_files(src, suffix='.npz')
    
    model_filename = pathlib.Path(m_src).stem
    data_dirname = pathlib.Path(src).stem
    ext_result = '.json'
    ext_preds = '.pkl'
    dst_main = os.path.join(dst, model_filename + '-' + transformation+ '-' + data_dirname)
    out = []
    out_pred_test = []
    
    
    
    for scr_file in tqdm(cv_scr_list):
        print(f"For {scr_file}:")
        src_file = pathlib.Path(scr_file)
        fold_name = src_file.stem
        
        data = load_data(str(scr_file))
        data_tensor, data_tensor_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
        
        if transformation == 'tucker':
            rank = -1
            data_tensor, data_tensor_test = get_tucker_tensors(data_tensor, data_tensor_test, rank=rank)
        elif transformation == 'svd':
            rank = -1
            data_tensor, data_tensor_test = get_SVD_tensors(data_tensor, data_tensor_test, rank=rank)
        else:
            pass

            
            

        
        init_model_dict = load_json(m_src)
        model = create_model(init_model_dict)
        
        print(model.name + ':')
        print(model.main_dict)
        print()
        
        # train
        model.fit(data_tensor, y_train)
        fitting_time = model.fit_time
        #test
        preds = model.predict(data_tensor_test)
        acc = accuracy_score(y_test, preds)
        print('Test acc = ', acc)
        #saving
        out_pred_test += [{'model': str(model) + fold_name, 'preds': preds, 'y_test': y_test}]
        save_pickle(out_pred_test, dst_main + ext_preds)
        
        out_dict = {'model': str(model) + fold_name,'accs': None, 'inf_time': None, 'mean_acc': acc, 'std_acc': None,
                    'fitting_time': fitting_time}
        out += [out_dict]
        save_json(out, dst_main + ext_result)
    
    return  out
         
    
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-m', '--m_src', default=None)
    
    parser.add_argument('-d', '--dst', default=None)
    parser.add_argument('-t', '--transform', default='origin')
    
    # parser.add_argument('-r', '--rank',  default=None,
    #                     help="For example rank='[1,2]' or rank = '1', default: None")
    
    arg = parser.parse_args()
    
    dst = pathlib.Path(arg.dst)
    assert dst.is_dir(), f'Error! {dst} should be dirrectory'
    classes = get_classes_dict()
    
     
    assert arg.transform.lower() in ['origin', 'tucker', 'svd'], "Error! -t should be one of"\
                                                              f" {['origin', 'tucker', 'svd']}, given {arg.transform.lower()}"
    
    
            
    work_with_model_cv(arg.src, arg.m_src, arg.dst, arg.transform, classes,)
    exit(0)