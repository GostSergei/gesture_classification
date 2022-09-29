# Conduction the first experiment
# parameters: 
# -s or --srs - a source of the data, file path.
# -m or --src_m - a source of the ML model description, file path. gesture_classification/stages/models/*
# -d or --dst - output dir of the model (dir path), output name will be formed automatically:
#     .json - accuracy data, 
#     .pkl - dictionary of {   y_test: ... , preds: ...}, 
#     .jpg - confusion matrix


import argparse
import os
import sys
import copy
import numpy as np
import pathlib
import json 
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# you need to install your abs path the the parent folder of modules
sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
from modules.models_module import create_model
from modules.bullets import get_classes_dict, load_data, load_json, save_json, save_pickle, decorator_script_wrap
from modules.ml_auxiliary import plot_confusion_matrix

        
@decorator_script_wrap        
def work_with_model_v2(src, m_src, dst, classes='def'):
    if classes == 'def':
        classes = get_classes_dict()
        
    data = load_data(src)
    data_tensor, data_tensor_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    
    init_model_dict = load_json(m_src)
    
    
    model_filename = pathlib.Path(m_src).stem
    data_filename = pathlib.Path(src).stem
    ext_result = '.json'
    ext_preds = '.pkl'
    ext_fig = '.jpg'
    dst_main = os.path.join(dst, model_filename+ '-' + data_filename)
        
   
    model = create_model(init_model_dict)
    
    print(model.name + ':')
    print(model.main_dict)
    print()
    
    # train
    model.fit(data_tensor, y_train)
    fitting_time = model.fit_time
    #test
    preds = model.predict(data_tensor_test)
    print('Test acc = ', accuracy_score(y_test, preds))
    save_pickle({'preds': preds, 'y_test': y_test}, dst_main + ext_preds)
    plot_confusion_matrix(data['y_test'], preds,  classes)
    plt.savefig(dst_main + ext_fig, dpi=300)
    print()
    # inferance
    inf_time = model.eval_inference_time(data_tensor_test[0:1], 100)
    print()
    # acc
    accs = model.eval_model(data_tensor, y_train, data_tensor_test, y_test)
    print(accs)
    out_dict = {'model': str(model),'accs': list(accs), 'inf_time': inf_time, 'mean_acc': accs.mean(), 'std_acc': accs.std(ddof=1),
                'fitting_time': fitting_time}
    
    save_json(out_dict, dst_main + ext_result)
    
    return  out_dict
         
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-m', '--m_src', default=None)
    
    parser.add_argument('-d', '--dst', default=None)
    # parser.add_argument('-r', '--rank',  default=None,
    #                     help="For example rank='[1,2]' or rank = '1', default: None")
    
    arg = parser.parse_args()
    
    dst = pathlib.Path(arg.dst)
    assert dst.is_dir(), f'Error! {dst} should be dirrectory'
                
    script_name = pathlib.Path(__file__).name
    
    print(f"Start {script_name}:")
    # script function
    work_with_model_v2(src=arg.src,m_src=arg.m_src, dst=arg.dst)
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()    