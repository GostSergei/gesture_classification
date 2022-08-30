import os
# import time_cropping_module as tcm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




SKELETON_KEY = 'skeleton'
INFO_KEY = 'info'
dict_sep='__'

ALL_DS_TYPES = ['flat02', 'deep']
ALL_HANDS = ['left', 'right', 'both']
LEN_GESTURES = 37
LEN_SUBS = 21
SEP_FLAT02 = '_'


SE_KEY = '_start_end_'
START_SE_KEY = 'action_start'
END_SE_KEY = 'action_end'




def get_all_gestures_from_DS_flat02(dataset_data_path, subs_name="02.G101-parsed"):
    sep = '_'
    aim_path = os.path.join(dataset_data_path, subs_name)

    # getting all dir names
    dirs = get_dirs(aim_path)
    # rez = os.listdir(aim_path)
    # dirs = []
    # for obj in rez:
    #     obj_path = os.path.join(aim_path, obj)
    #     if os.path.isdir(obj_path):
    #         dirs += [obj]

    # getting all gestures
    gestures = set()
    for dir in dirs:
        gesture = '_'.join(dir.split(sep)[0:-2])
        gestures.add(gesture)
    gestures = list(gestures)
    return gestures



def get_ds_main_parameters(ds_type):
    if ds_type == 'flat02':
        subs_name_fstr= "02.{}-parsed"
        trial_name_fstr = "trial{}"
        data_file_name = 'joints.csv'
    else:
        raise(Exception('Error! ds_type = {} is not realized yet'))


    return subs_name_fstr, trial_name_fstr, data_file_name


def get_dirs(path):
    rez = os.listdir(path)
    dirs = []
    for obj in rez:
        obj_path = os.path.join(path, obj)
        if os.path.isdir(obj_path):
            dirs += [obj]
    return dirs


# Checking data 
def check_ds_dict(ds_dict, verbose=True, skeleton_key=SKELETON_KEY, info_key=INFO_KEY):
    count_info = 0
    count_skelet = 0
    keys_number = len(ds_dict.keys())
    bad_ids = []
    for id in ds_dict.keys():
        f_info, f_skelet = True, True
        for key in ds_dict[id]:
            if key == info_key:
                count_info += 1
                f_info = False
            if key == skeleton_key:
                count_skelet += 1
                f_skelet = False
                
        if f_info or f_skelet:
                bad_ids += [id]
    good = count_info == count_skelet == keys_number
    if verbose:
        print(good, f"\tInfo: {count_info};   Skeleton: {count_skelet};   Total: {keys_number}")
        if not good:
            print(f"{len(bad_ids)/2}({keys_number - count_info}\{keys_number - count_skelet})")
    return good

def check_ds_dict_with_se(ds_dict, verbose=True, skeleton_key=SKELETON_KEY, info_key=INFO_KEY, se_key=SE_KEY):
    count_info = 0
    count_skelet = 0
    count_se = 0
    keys_number = len(ds_dict.keys())
    bad_ids, bad_se_ids = [], []
    for id in ds_dict.keys():
        f_info, f_skelet, f_se = True, True, True
        for key in ds_dict[id]:
            if key == info_key:
                count_info += 1
                f_info = False
            if key == skeleton_key:
                count_skelet += 1
                f_skelet = False
            if key == se_key:
                count_se += 1
                f_se = False
                
        if f_info or f_skelet:
                bad_ids += [id]
        if f_se:
            bad_se_ids += [id]
    good = count_info == count_skelet == keys_number
    good_se = count_se == keys_number
    if verbose:
        print(good, f"\tInfo: {count_info};   Skeleton: {count_skelet}; Total: {keys_number}")
        print(good_se, f"\t SE: {count_se}   Total: {keys_number}")
        if not good:
            print(f"{len(bad_ids)/2}({keys_number - count_info}\{keys_number - count_skelet})")
        if not good_se:
            print(f"{len(bad_se_ids)/1}({keys_number - count_se})")

    return good

def check_ds_se_info(ds_dict, verbose=True, info_key=INFO_KEY, end_key=END_SE_KEY, start_key=START_SE_KEY,se_key=SE_KEY, label_key='label', valid_key='is_valid_performance'):
    label_key = 'label'
    count_start = 0
    count_end = 0
    count_valid = 0
    keys_number = len(ds_dict.keys())
    bad_ids =  []
    for g_id in ds_dict.keys():
        keys = ds_dict[g_id][info_key][label_key].keys()

        f_valid, f_start, f_end = True, True, True
        if valid_key in keys:
            count_valid += 1
            f_valid = False

        if end_key in keys:
            count_end += 1
            f_end= False

        if start_key in keys:
            count_start += 1
            f_start = False

                
        if f_valid or f_start or f_end:
                bad_ids += [id]
 
    good = count_start == count_end ==  count_valid == keys_number
    if verbose:
        print(good, f"\tValid: {count_valid};   Start: {count_start};   End: {count_end};   Total: {keys_number}")

        if not good:
            print(f"{len(bad_ids)}({keys_number - count_valid}\{keys_number - count_start}\{keys_number - count_end})")

    return good





def get_gesture_from_g_id(g_id):
    sep = '__'
    g_sep = '_'
    gesture = g_id.split(sep)[3:]
    gesture = g_sep.join(gesture)
    return gesture

def get_sub_from_g_id(g_id):
    sep = '__'
    sub = g_id.split(sep)[0]
    return sub

def get_hand_from_g_id(g_id):
    sep = '__'
    hand = g_id.split(sep)[2]
    return hand

def get_trial_name_from_g_id(g_id):
    sep = '__'
    trial_name = g_id.split(sep)[1]
    return trial_name

def _get_all_elements_from_ds_dict_(ds_dict, get_element_fun):
    elements = set()
    for g_id in ds_dict.keys():
        element =  get_element_fun(g_id)
        elements.add(element)
    elements = list(elements)
    elements.sort()
    return elements


def get_all_subs_from_ds_dict(ds_dict):
    return _get_all_elements_from_ds_dict_(ds_dict, get_sub_from_g_id)

def get_all_gestures_from_ds_dict(ds_dict):
    return _get_all_elements_from_ds_dict_(ds_dict, get_gesture_from_g_id)

def get_all_trial_names_from_ds_dict(ds_dict):
    return _get_all_elements_from_ds_dict_(ds_dict, get_trial_name_from_g_id)

def get_all_hands_from_ds_dict(ds_dict):
    return _get_all_elements_from_ds_dict_(ds_dict, get_hand_from_g_id)


def form_g_id(sub, trial_name, hand, gesture):
    dict_sep = '__'
    g_id = dict_sep.join([sub, trial_name, hand, gesture])
    return g_id






def join_function_v0(se_array, axis, k=1.7):
    if se_array.shape[0] == 1:
        return se_array.reshape(-1)

    # print(se_array)
        
    # means = se_array.mean(axis=axis)
    # stds = se_array.std(axis=axis)

    # # too far from std
    # mask_std = np.abs(se_array - se_array.mean(axis=axis)) < se_array.std(axis=axis)*k
    # print(mask_std)
    # print(stds)
    # se_array = se_array[np.all(mask_std == True, axis=1, keepdims=True)]
    # print(se_array)
    se = np.mean(se_array, axis=axis)
    return se



def clear_key_from_ds_dict(ds_dict, clear_key):
    for g_id in ds_dict.keys():
        if clear_key in ds_dict[g_id]:
            del ds_dict[g_id][clear_key]
    return ds_dict

def check_info_label(ds_dict, g_id):
    label = ds_dict[g_id][INFO_KEY]['label']
    return label






# # from IPython.display import clear_output
# def add_se_to_ds_dict(ds_dict, target_subs, target_hands, target_gestures, target_trial_names=None, get_crop_fun=tcm.get_crop_points_by_diff, crop_args={'threshold':0.15}, join_function=join_function_v0,
#                       main_points_r = [14, 16], main_points_l = [13, 15], main_coords=['x', 'y']):
#     for sub in target_subs:
#         for hand in target_hands:
#             print()
#             print(f"For {sub} and {hand} hand:")
#             for gesture in target_gestures:
#                 print(gesture, end=';  ')

#                 if hand == 'right':
#                     main_points = main_points_r
#                 elif hand == 'left':
#                     main_points = main_points_l
#                 elif hand == 'both':
#                     main_points = main_points_l + main_points_r
#                 points = [f"pose__{num}__{coord}" for num in main_points for coord in main_coords]

                
#                 f_new_trial, trial = True, 0
#                 while f_new_trial:
#                     trial += 1
#                     trial_name = f'trial{trial}'

#                     g_id = form_g_id(sub, trial_name, hand, gesture)
#                     if g_id not in ds_dict.keys():
#                         f_new_trial = False
#                         continue
                    
#                     if target_trial_names is not None:
#                         if trial_name not in target_trial_names:
#                             continue

                    
#                     signals = ds_dict[g_id][SKELETON_KEY][points]
#                     # Adwanced usage
#                     s_e = tcm.get_cropping_points(signals, around=False, get_crop_fun=get_crop_fun, crop_args=crop_args, join_function=join_function)
#                     ds_dict[g_id][SE_KEY] = s_e
               


#     # clear_output()
#     print()
#     print()
#     trial_str = 'all trials'
#     if target_trial_names is not None:
#         trial_str = target_trial_names
#     print(f"Adding se for \nsubs:{target_subs} \nhands:{target_hands} \ngestures:{target_gestures} \ntrials: {trial_str} \nwas done!") 
#     return ds_dict




# def show_signals(ds_dict, g_id, points):
#     signals = ds_dict[g_id][SKELETON_KEY][points]
#     s_e = ds_dict[g_id][SE_KEY]
#     columns = list(signals.columns)
#     for i, column in enumerate(columns):
#         s = signals[column]
#         s = s - s[0]
#         # s = (s - s[0]).abs()
#         s = s/s.abs().max()

#         if i == 0:
#             tcm.plot_cropping(s, s_e, title=g_id, label=column, n_diff=0)
#         else:
#             plt.plot(s, label=column)
#     plt.legend()

    

    
