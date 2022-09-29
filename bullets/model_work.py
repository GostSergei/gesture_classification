# the old vertion of the work with models!

import argparse
import os
import sys
import copy
import numpy as np
import pathlib
import json 
from sklearn.metrics import accuracy_score

# you need to install your abs path the the parent folder of modules
sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
from modules.models_module import create_model
from modules.bullets import load_data, ask_confirmation, load_json, decorator_script_wrap

 
@decorator_script_wrap
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
    inf_time = model.eval_inference_time(data_tensor_test[0:1], 100)
    print()
    # acc
    accs = model.eval_model(data_tensor, y_train, data_tensor_test, y_test)
    print(accs)
    out_dict = {'model': str(model),'accs': list(accs), 'inf_time': inf_time, 'mean_acc': accs.mean(), 'std_acc': accs.std(ddof=1)}
    with open(dst, 'w') as f:
        json.dump(out_dict, f)
    return  out_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', default=None)
    parser.add_argument('-m', '--m_src', default=None)
    parser.add_argument('-d', '--dst', default=None)

    arg = parser.parse_args()
    
    if not pathlib.Path(arg.dst).is_dir():
        if pathlib.Path(arg.dst).exists():
            if not ask_confirmation(f'Reload {arg.dst}?'):
                print('Script was stopped!')
                exit(0)
            else:
                print(f'{arg.dst} will be reloaded')
                
    script_name = pathlib.Path(__file__).name
    
    print(f"Start {script_name}:")
    # script function
    work_with_model(src=arg.src, m_src=arg.m_src, dst=arg.dst)
    ###
    print(f"Finished script: {script_name}")
    print()
    

if __name__ == '__main__': 
    main()