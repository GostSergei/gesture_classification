import os
import pathlib
import pandas as pd

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


def get_fancy_table(table):
    mean_col = (100*table['mean_acc']).round(2).map(lambda x: str(x))
    std_col = (100*table['std_acc']).round(2).map(lambda x: f"({x})" if str(x) != 'nan' else '')
    table['Accuracy, %'] = mean_col + std_col
    table['Inference time, ms'] = (table['inf_time']*1000).round(3).map(lambda x: str(x) + '')
    return table[['model', 'Accuracy, %', 'Inference time, ms']]

