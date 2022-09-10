import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        # pref = [pref] if pref != '' else []
        
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
        name = res[name_key].split("-")[-1]
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
    
    
        
        
    

