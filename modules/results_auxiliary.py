import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.metrics import accuracy_score

try:
    from modules.bullets import load_json 
except:
    pass

try:
    from bullets import load_json 
except:
    pass



def get_target_files(src, suffix='.json'):
    src = pathlib.Path(src)
    src_list = []
    if src.is_file():
        src_list += [str(src)]
    
    if src.is_dir():
        for src_file in os.listdir(src):
            if pathlib.Path(src_file).suffix == suffix:
                src_list += [os.path.join(src, src_file)]
    return src_list

# first version of results 
def form_table_v1(src_list, dst=None):
    
    df = pd.DataFrame()
    for src_file in src_list:
        res_dict = load_json(src_file)
        main_dict = {}
        
        
        
        for key in ['model','mean_acc', 'std_acc', 'inf_time']:
            main_dict[key] = [res_dict[key]]
           
        file_name = pathlib.Path(src_file).stem
        pref = '_'.join(file_name.split('_')[1:])
        if pref == '':
            pref = 'original'
        # pref = [pref] if pref != '' else []
        
        main_dict['model'] = '+'.join([pref] + [res_dict['model']])
        df_row = pd.DataFrame(main_dict)
        df = pd.concat([df, df_row])
        
    df = df.sort_values(['model'])
    
    df = df.reset_index(drop=True)
    return df


# the main version of results for the stage 2 
def form_table_v2(src_list, dst=None):
    df = pd.DataFrame()

    for src_file in src_list:
        res_dict = load_json(src_file)
        main_dict = {}
        
        for key in ['model','mean_acc', 'std_acc', 'inf_time', 'fitting_time']:
            if key in res_dict.keys():
                main_dict[key] = [res_dict[key]]
            else:
                print(f'Warning! {key} is not in data')
           
        file_name = pathlib.Path(src_file).stem
        pref = '_'.join(file_name.split('_')[1:])
        if pref == '':
            pref = 'original'
        
        main_dict['model'] = '+'.join([pref] + [res_dict['model']])
        df_row = pd.DataFrame(main_dict)
        df = pd.concat([df, df_row])
        
    df = df.sort_values(['model'])
    
    df = df.reset_index(drop=True)
    return df


# the main version of results for the stage 3 
def form_table_v3(src_list, dst=None):
    
    df = pd.DataFrame()

    for src_file in src_list:
        res_dict = load_json(src_file)
        main_dict = {}
        
        for key in ['model','mean_acc', 'std_acc', 'inf_time', 'fitting_time']:
            if key in res_dict.keys():
                main_dict[key] = [res_dict[key]]
            else:
                print(f'Warning! {key} is not in data')
           
        file_name = pathlib.Path(src_file).stem
        pref = '-'.join(file_name.split('-')[-1])
        
        main_dict['model'] = '+'.join([pref] + [res_dict['model']])
        df_row = pd.DataFrame(main_dict)
        df = pd.concat([df, df_row])
        
    df = df.sort_values(['model'])
    
    df = df.reset_index(drop=True)
    return df





def get_fancy_table_v1(table):
    mean_col = (100*table['mean_acc']).round(2).map(lambda x: str(x))
    std_col = (100*table['std_acc']).round(2).map(lambda x: f"({x})" if str(x) != 'nan' else '')
    table['Accuracy, %'] = mean_col + std_col
    table['Inference time, ms'] = (table['inf_time']*1000).round(3).map(lambda x: str(x) + '')
    
    return table[['model', 'Accuracy, %', 'Inference time, ms']]

def get_fancy_table_v2(table):
    mean_col = (100*table['mean_acc']).round(2).map(lambda x: str(x))
    std_col = (100*table['std_acc']).round(2).map(lambda x: f"({x})" if str(x) != 'nan' else '')
    table['Accuracy, %'] = mean_col + std_col
    table['Inference time, ms'] = (table['inf_time']*1000).round(3).map(lambda x: str(x) + '')
    if 'fitting_time' in table.columns:
        table['Fitting time, s'] = (table['fitting_time']).round(2).map(lambda x: str(x) + '')
    
    return table[['model', 'Accuracy, %', 'Inference time, ms', 'Fitting time, s']]





def collect_accuracy(src, value_key='mean_acc', name_key = 'model'):
    res_list = load_json(src)
    model_name = res_list[0][name_key].split(":")[0]
    acc_dict = {}
    for res in res_list:
        name = res[name_key].split("-")[1]
        acc_dict[name] = [res[value_key]]
    acc_dict = dict(sorted(acc_dict.items()))
    return acc_dict, model_name


def form_common_table(acc_dicts, model_names, table=None, name_dict=None):
    acc_col, model_col = 'Accuracy', 'Model'
    if table is None:
        table = pd.DataFrame(columns=[acc_col, model_col])
        
    if name_dict is None:
        name_dict = {}
        
    for model_name in model_names:
        if model_name not in name_dict.keys():
            name_dict[model_name] = model_name
        
        
    for acc_dict, model_name in zip(acc_dicts, model_names):
        df = pd.DataFrame(acc_dict, index=[acc_col]).T
        df[model_col] = name_dict[model_name]
        table = pd.concat([table, df])
    return table

def plot_swarmboxplot(table, y='Accuracy', x='Model', ax=None):
    sns.swarmplot( x=x, y=y, data=table, ax=ax)
    sns.boxplot(x=x, y=y, data=table, showcaps=True, boxprops={'facecolor':'None'},
                    showfliers=True,whiskerprops={'linewidth':0.75}, ax=ax)
    
    
    

        
        
def  drop_gesture(preds_in, y_test_in, drop_classes=[16]):
    preds,  y_test = np.copy(preds_in), np.copy(y_test_in)
    
    for label in drop_classes:
        delete_idxs = np.where(y_test == label)
        y_test = np.delete(y_test, delete_idxs)
        preds = np.delete(preds, delete_idxs)
        
    return preds,  y_test

def  merge_gesture(preds_in, y_test_in, merge_classes=[[17, 16]]):
    preds,  y_test = np.copy(preds_in), np.copy(y_test_in)
    
    for label_pair in merge_classes:
        # update_idxs = np.where(y_test == label_pair[1])
        update_value =  label_pair[0]
        y_test[np.where(y_test == label_pair[1])] = update_value
        preds[np.where(preds == label_pair[1])] = update_value
        
    return preds,  y_test




def filter_acc(src, drop_classes=[16], merge_classes=[[17, 16]]):
    df = pd.DataFrame()

    # src_list_pkl = get_target_files(src, suffix='.pkl')
    src_list_json = get_target_files(src, suffix='.json')
    
    for file_json in src_list_json:
        
        file_pkl = file_json[:-5] + '.pkl'
        
        main_dict = {}
        file_name = pathlib.Path(file_json).stem
        pref = '_'.join(file_name.split('_')[1:])
        if pref == '':
            pref = 'original'
        # pref = [pref] if pref != '' else []

        res_dict = load_json(file_json)
        main_dict['model'] = '+'.join([pref] + [res_dict['model']])
        
        with open(file_pkl, 'rb') as f:
            data = pickle.load(f)

        preds, y_test = data['preds'], data['y_test']
        
        preds_drop, y_test_drop = drop_gesture(preds, y_test, drop_classes)
        acc_drop = accuracy_score(y_test_drop, preds_drop)
        main_dict['acc_drop'] = [acc_drop]
        main_dict['acc_drop_fancy'] = np.round(100*acc_drop, 2)
        
        preds_merge, y_test_merge = merge_gesture(preds, y_test, merge_classes)
        acc_merge = accuracy_score(y_test_merge, preds_merge)
        main_dict['acc_merge'] = [acc_merge]
        main_dict['acc_merge_fancy'] = np.round(100*acc_merge, 2)
        
        
        
        df_row = pd.DataFrame(main_dict)
        df = pd.concat([df, df_row])
        
        
    df = df.sort_values(['model'])
    df = df.reset_index(drop=True)
    return df
 
        
     
    
    
        
        
    

