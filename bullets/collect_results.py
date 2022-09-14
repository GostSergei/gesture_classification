# import argparse
# import os
# import sys
# import copy
# import numpy as np
# import pathlib
# import json
# import pickle

# # you need to install your abs path the the parent folder of modules
# sys.path.append('/home/s.gostilovich/gesture_progect/gesture_classification')
# from modules.classification_auxiliary import get_tucker_tensors

# def load_data(src):
#     data_ = np.load(src, allow_pickle=True)
#     nan = 0
#     data = {}
#     for key in ['x_train', 'x_test', 'y_train', 'y_test']:
#         data[key] = data_[key]
#         if np.isnan(data[key]).sum() > 0:
#             data[key] = np.nan_to_num(data[key], nan=nan, posinf=nan)
#             print(f'For {key} nan will be replaced by {nan}!')
#     return data

    
# def load_json(src):
#     with open(src, 'r') as f:
#         out = json.load(f)
#     return out


# def save_json(data, dst):
#     with open(dst, 'w') as f:
#         json.dump(data, f)
            
# def save_pickle(data, dst):
#     with open(dst, 'wb') as f:
#         pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
        
    
    
# # def tucker_decomposition(src, dst, rank=None):
# #     data = load_data(src)
# #     data_tensor, data_tensor_test = data['x_train'], data['x_test']
# #     tensor_tucker, tensor_tucker_test = get_tucker_tensors(data_tensor, data_tensor_test)
# #     data_tucker = copy.deepcopy(data)
# #     data_tucker['x_train'] = tensor_tucker
# #     data_tucker['x_test'] = tensor_tucker_test
    
# #     if dst is not None:
# #         with open(dst, 'wb') as f:
# #             np.save(f, data_tucker)
            
#     # return data_tucker


# def ask_confirmation(question_text,):
#     out = None
#     while out is None:
#         str_ = input(question_text + ' [y/n]: ')
#         if str_.lower() in ['y', 'yes']:
#             out = True
#         elif str_.lower() in ['n', 'no']:
#             out = False
#         else:
#             print(f"{str_} is not valid anwser!" + ' [y/n]: ')
#     return out
        

# def get_target_files(src, suffix='.json'):
#     src = pathlib.Path(src)
#     src_list = []
#     if src.is_file():
#         src_list += [str(src)]
    
#     if src.is_dir():
#         for src_file in os.listdir(src):
#             if pathlib.Path(src_file).suffix == suffix:
#                 src_list += [str(src_file)]
#     return src_list

# def form_table(src_list, dst=None):
    
#     pass
    

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s', '--src', default=None)
#     parser.add_argument('-d', '--dst', default=None)
    
#     arg = parser.parse_args()
    
#     dst = pathlib.Path(arg.dst)
#     if pathlib.Path(arg.dst).exists():
#         if not ask_confirmation(f'Reload {arg.dst}?'):
#             print('Script was stopped!')
#             exit(0)
#         else:
#             print(f'{arg.dst} will be reloaded')
            
#     assert dst.is_file(), 'Error! {dst} should be a csv file!'
            
#     src_list = get_target_files(arg.src)
#     df = form_table(src_list, dst)
    
            
#     # tucker_decomposition(arg.src, arg.dst, arg.rank)
#     exit(0)