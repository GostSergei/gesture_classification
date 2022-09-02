
import copy

import tensorly as tl
# from dataset_auxiliary import *
from modules.dataset_auxiliary import *


def select_valid_samplels(ds_dict):
    data_dict = copy.deepcopy(ds_dict)
    len_ds = len(ds_dict.keys())
    for g_id in ds_dict.keys():
        is_valid = ds_dict[g_id][INFO_KEY]['label']['is_valid_performance']
        if not is_valid:
            del data_dict[g_id]
      
    len_ds = len(ds_dict.keys())      
    len_data = len(data_dict.keys())
    print(f"{len_ds - len_data} samples was unvalied. Ramain {len_data}({len_ds})") 
            
    return data_dict

def select_g_ids(input_data_dict, subs=None, gestures=None, trials=None, hands=None):
    data_dict = {}
    if subs == None:
        subs = get_all_subs_from_ds_dict(input_data_dict)
    if gestures == None:
        gestures = get_all_gestures_from_ds_dict(input_data_dict)
    if trials == None:
        trial_names = get_all_trial_names_from_ds_dict(input_data_dict)
    else:
        trial_names = [f"trail{trial}" for trial in trials]
        
    if hands == None:
        hands = get_all_hands_from_ds_dict(input_data_dict)
        
    for sub in subs:
        for trial_name in trial_names:
            for gesture in gestures:
                for hand in hands:
                    # print(sub, trial_name, gesture, hand)
                    g_id = form_g_id(sub, trial_name, hand, gesture)
                    if g_id in input_data_dict.keys():
                        data_dict[g_id] = copy.deepcopy(input_data_dict[g_id])
                        
    len_in = len(input_data_dict.keys())      
    len_out= len(data_dict.keys())
    print(f"Selected  {len_out} samples. {len_in - len_out}({len_in}) was dropped!") 
                    
    return data_dict


def form_important_points(df_skeleton, coords=['x', 'y', 'z'], pose_points='def', hand_points='def'):
    # for pose
    df = df_skeleton
    coords = ''.join(coords)
    if pose_points == 'def':
        pose_points = [i for i in range(25)]
    if hand_points == 'def':
        hand_points = [i for i in range(21)]
        
        
        
    pose_points = '|'.join([str(i) for i in pose_points])
    cols_pose = df.columns.str.match(f"^pose__({pose_points})__[{coords}]")
    cols_pose = list(df.columns[cols_pose])

    hand_points = '|'.join([str(i) for i in hand_points])
    cols_hands = df.columns.str.match(f"^(right|left)_hand__({hand_points})__[{coords}]")
    cols_hands = list(df.columns[cols_hands])
    
    cols = cols_pose + cols_hands
    df = df[cols]
    return df    



def update_skeleton(input_data_dict, update_df_func, kwargs={}, show=True):
    data_dict = copy.deepcopy(input_data_dict)
    delta_list = []
    for g_id in data_dict.keys():
        df = data_dict[g_id][SKELETON_KEY]
        df, delta = update_df_func(df, **kwargs)
        data_dict[g_id][SKELETON_KEY] = df
        delta_list += [delta]
        
    if show:
        plt.plot(delta_list)
        plt.title('Delta')
        plt.gcf().set_size_inches((6, 1))
        
    return data_dict       


def select_points(input_data_dict, coords=['x', 'y', 'z'], pose_points='def', hand_points='def', show=True):
    kwargs = {'coords':coords, 'pose_points':pose_points, 'hand_points':hand_points,}
    data_dict = update_skeleton(input_data_dict, update_df_select_points, kwargs, show)
    
    if show:
        plt.title('Delta len for points')
        
    return data_dict


def update_df_fill_na(df, value=None, method=None, axis=None, limit=None):
    delta = df.isna().sum()
    df = df.fillna(value=value, method=method, axis=axis, limit=limit)
    return df, delta

def update_df_select_points(df, coords=['x', 'y', 'z'], pose_points='def', hand_points='def'):
    before = len(df.columns)
    df = form_important_points(df, coords, pose_points, hand_points)
    after = len(df.columns)
    return df, before - after 

def update_df_solve_time(df, max_time=120):
    delta = True
    df = df.iloc[:max_time, :]
    return df, delta





def get_class_name_from_g_id(g_id):
    sep = '__'
    gesture = get_gesture_from_g_id(g_id)
    hand =  get_hand_from_g_id(g_id)
    class_name = sep.join([hand, gesture])
    return class_name
    
CLASS_NAME_KEY = 'class_name'
CLASS_LABEL_KEY = 'class_label'


def form_gesture_labels(data_dict):
    name_label_dict = {}
    current_label = 0
    for g_id in data_dict.keys():
        class_name = get_class_name_from_g_id(g_id)
        if class_name not in name_label_dict.keys():
            name_label_dict[class_name] = current_label
            current_label += 1
                   
        data_dict[g_id][CLASS_NAME_KEY] = class_name
        data_dict[g_id][CLASS_LABEL_KEY] = name_label_dict[class_name]
        
        
    # switching keys and values
    label_dict = {y: x for x, y in name_label_dict.items()}
    return data_dict, label_dict


def form_samples_labels_lists(data_dict):
    samples_list,  labels_list = [], []
    for g_id in data_dict.keys():
        labels_list += [data_dict[g_id][CLASS_LABEL_KEY]]
        data = data_dict[g_id][SKELETON_KEY].to_numpy()
        samples_list += [np.expand_dims(data, 0)]
    return samples_list,  labels_list




# Tensors
def gen_eye_core_tensor(rank, n_dims):
    core_t = tl.zeros(shape=[rank for i in range(n_dims)])
    for i in range(rank):
        idxs = [i for j in range(n_dims)]
        exec(f"core_t{idxs} = 1" )        
    return core_t

def gen_diag_core_tensor(rank, n_dims, diags=1):
    if isinstance(diags, int):
        diags = [diags for i in range(rank)]
    core_t = tl.zeros(shape=[rank for i in range(n_dims)])
    for i in range(rank):
        idxs = [i for j in range(n_dims)]
        exec(f"core_t{idxs} = {diags[i]}" )     
        
    return core_t



def show_decomposition_error(sigma_array, norm_t,   title='', start=1, end=None, version=2, show=True):
    if end is None:
        end = len(sigma_array)
        
    if version == 1:
        plt.plot((np.cumsum((sigma_array[::-1]/norm_t)**2)[::-1])[start:end])
    elif version == 2:
        plt.plot((1 - np.cumsum((sigma_array/norm_t)**2))[start-1:end])
    else:
        print('Bad verion!')
        return
    plt.title(title)
    plt.grid()
    plt.gcf().set_size_inches((16, 2))
    if show:
        plt.show()
    return 



def recover_svd(u_s_v, rank=None):
    u, s, v = u_s_v
    if rank is None:
        rank = len(s)
    if len(s) < rank:
        print(f"Worning! Rank [{rank}] is more then len(sigmas) [{len(s)}]!")
        rank = len(s)
        
    recover_mat = np.matmul(u[:, :rank], np.diag(s[:rank]))
    recover_mat = np.matmul(recover_mat, v[:rank, :])
    return recover_mat



        
        
    
    