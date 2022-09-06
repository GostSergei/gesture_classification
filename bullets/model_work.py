import argparse
import os
import sys
import copy
import numpy as np
import pathlib
import json 
from sklearn.metrics import accuracy_score
sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
from modules.models_module import create_model
# import modules.models_module 


def load_data(src):
    data = np.load(src, allow_pickle=True)
    print(type(data))
    return data
    print(type(data))
    return {
        
        'x_train': np.nan_to_num(data['x_train'], nan=0, posinf=0),
        'x_test': np.nan_to_num(data['x_test'], nan=0, posinf=0),
        'y_train': np.nan_to_num(data['y_train'], nan=0, posinf=0),
        'y_test': np.nan_to_num(data['y_test'], nan=0, posinf=0),
    }
    
def load_json(src):
    with open(src, 'r') as f:
        out = json.load(f)
    return out
        
    
    



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
        
        
def work_with_model(src, m_src, dst):
    data = load_data(src)
    data_tensor, data_tensor_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
    
    init_model_dict = load_json(m_src)
    
    if not pathlib.Path(dst).is_file():
        model_filename = pathlib.Path(m_src).stem
        data_filename = pathlib.Path(src).stem
        ext = '.json'
        dst = os.path.join(dst, model_filename+ '-' + data_filename + ext)
        
   
    model = create_model(init_model_dict)
    
    print(model.name + ':')
    print(model.main_dict)
    print()
    
    # train
    model.fit(data_tensor, y_train)
    #test
    preds = model.predict(data_tensor_test)
    print('Test acc = ', accuracy_score(y_test, preds))
    print()
    # inferance
    inf_time = model.eval_inference_time(data_tensor_test[0:1,:,:], 100)
    print()
    # acc
    accs = model.eval_model(data_tensor, y_train, data_tensor_test, y_test)
    print(accs)
    out_dict = {'model': str(model),'accs': list(accs), 'inf_time': inf_time, 'mean_acc': accs.mean(), 'std_acc': accs.std(ddof=1)}
    with open(dst, 'w') as f:
        json.dump(out_dict, f)
    return  out_dict
        
    
    
    
    
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-m', '--m_src', default=None)
    parser.add_argument('-d', '--dst', default=None)
    
    parser.add_argument('-r', '--rank',  default=None,
                        help="For example rank='[1,2]' or rank = '1', default: None")
    
    arg = parser.parse_args()
    
    dst = pathlib.Path(arg.dst)
    if not pathlib.Path(arg.dst).is_dir():
        if pathlib.Path(arg.dst).exists():
            if not ask_confirmation(f'Reload {arg.dst}?'):
                print('Script was stopped!')
                exit(0)
            else:
                print(f'{arg.dst} will be reloaded')
            
    work_with_model(arg.src, arg.m_src, arg.dst)
    exit(0)