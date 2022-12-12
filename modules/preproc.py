from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

import plotly.graph_objects as go



# loading





# pose points 
MAIN_BODY_POINTS = list(range(0, 23))
POSE_POINTS = list(range(0, 33))
HANDS_LINE_POINTS = [21, 15, 17, 19, 15, 13, 11, 12, 14, 16, 20, 18, 16, 22]
FACE_LINE_POINTS = [0, 10, 9, 0, 8, 6, 5, 4, 0, 7, 3, 2, 1, 0]
LEGS_LINE_POINTS = [28, 32, 30, 28, 26, 24, 23, 25, 27, 29, 31, 27]
BOXBODY_LINE_POINTS = [11, 12, 24, 23, 11]
# POSE_LEFT_WRIST = [15]
# POSE_RIGHT_WRIST = [16]

POSE_ENUM =dict(NOSE = 0,
                LEFT_EYE_INNER = 1,
                LEFT_EYE = 2,
                LEFT_EYE_OUTER = 3,
                RIGHT_EYE_INNER = 4,
                RIGHT_EYE = 5,
                RIGHT_EYE_OUTER = 6,
                LEFT_EAR = 7,
                RIGHT_EAR = 8,
                MOUTH_LEFT = 9,
                MOUTH_RIGHT = 10,
                LEFT_SHOULDER = 11,
                RIGHT_SHOULDER = 12,
                LEFT_ELBOW = 13,
                RIGHT_ELBOW = 14,
                LEFT_WRIST = 15,
                RIGHT_WRIST = 16,
                LEFT_PINKY = 17,
                RIGHT_PINKY = 18,
                LEFT_INDEX = 19,
                RIGHT_INDEX = 20,
                LEFT_THUMB = 21,
                RIGHT_THUMB = 22,
                LEFT_HIP = 23,
                RIGHT_HIP = 24,
                LEFT_KNEE = 25,
                RIGHT_KNEE = 26,
                LEFT_ANKLE = 27,
                RIGHT_ANKLE = 28,
                LEFT_HEEL = 29,
                RIGHT_HEEL = 30,
                LEFT_FOOT_INDEX = 31,
                RIGHT_FOOT_INDEX = 32,
                ) 

# hand_points
# MAIN_BODY_POINTS = list(range(0, 23))
HAND_POINTS = list(range(0, 21))
THUMB_LINE_POINTS = [0, 1, 2, 3, 4]
INDEX_LINE_POINTS = [5, 6, 7, 8]
MIDDLE_LINE_POINTS = [9, 10, 11, 12]
RING_LINE_POINTS = [13, 14, 15, 16]
PINKY_LINE_POINTS = [17, 18, 19, 20]
# PALMBOX_LINE_POINTS = [0, 5, 9, 13, 17, 0]
PALMBOX_LINE_POINTS = [0, 5, 9, 13, 17, 0, 1, 5, 0]
PALMBOX_LINE_POINTS = [0, 1, 5, 9, 13, 17, 0]

HAND_ENUM = dict(WRIST = 0,
                THUMB_CMC = 1,
                THUMB_MCP = 2,
                THUMB_IP = 3,
                THUMB_TIP = 4,
                INDEX_FINGER_MCP = 5,
                INDEX_FINGER_PIP = 6,
                INDEX_FINGER_DIP = 7,
                INDEX_FINGER_TIP = 8,
                MIDDLE_FINGER_MCP = 9,
                MIDDLE_FINGER_PIP = 10,
                MIDDLE_FINGER_DIP = 11,
                MIDDLE_FINGER_TIP = 12,
                RING_FINGER_MCP = 13,
                RING_FINGER_PIP = 14,
                RING_FINGER_DIP = 15,
                RING_FINGER_TIP = 16,
                PINKY_MCP = 17,
                PINKY_PIP = 18,
                PINKY_DIP = 19,
                PINKY_TIP = 20,
                )



# Functions:
# 2) preprocess csv - points 

""" 
Points coordinates: xs, yx, zs
"""


# hands
def get_hand_points_coordinates(df, points, hand_type='left',  i_row='all',  to_np=True, coord_type='local'):
    # coord_types = ['local', 'global']
    coord_types = ['local']
    assert coord_type in coord_types, f"Error! Invalide coord_type={coord_type}. Possible values: {coord_types}"

    hand_types = ['left', 'l', 'right', 'r']
    assert hand_type in hand_types, f"Error! Invalide coord_type={hand_type}. Possible values: {hand_types}"

    if i_row == 'all':
        df = df.iloc[:,:]
    else:
         df = df.iloc[i_row:i_row+1,:]
            
    axis_names = ['x', 'y', 'z']
    x_y_zs = []

    if hand_type == 'r':
        hand_type = 'right'
    if hand_type == 'l':
        hand_type = 'left'

    if coord_type == 'local':
        coord_str = hand_type + '_hand'
    # elif coord_type == 'global':
    #     coord_str = 'pose_world'
    else:
        print('Warning! Bad coord_type!')
    
 
    for axis_name in axis_names:
        columns_i = [f"{coord_str}__{point}_{axis_name}" for point in points] 
        x_y_zs += [df[columns_i]]
        
    if to_np:
        for i in range(len(x_y_zs)):
            x_y_zs[i] = x_y_zs[i].to_numpy()
            if x_y_zs[i].shape[0] == 1:
                x_y_zs[i] = x_y_zs[i].reshape(-1)
                        
    xs, yx, zs = x_y_zs
    
    return xs, yx, zs



def get_hand_points_tensor(df_pose, points, hand_type='left', i_row='all', xyz_axis=0,  coord_type='local'):
    xs, yx, zs = get_hand_points_coordinates(df_pose, points, hand_type, i_row=i_row,  coord_type=coord_type,  to_np=True)
    tensor = np.stack([xs, yx, zs], axis=xyz_axis)
    return tensor

def get_hand_tensor(df_pose, hand_type='left', i_row='all', xyz_axis=0, coord_type='local'):
    points = list(range(0, 21))
    return get_hand_points_tensor(df_pose, points, hand_type,  i_row, xyz_axis=xyz_axis, coord_type=coord_type)




# pose
def get_pose_points_coordinates(df_pose, points, i_row='all',  to_np=True, coord_type='local'):

    coord_types = ['local', 'global']
    assert coord_type in coord_types, f"Error! Invalide coord_type={coord_type}. Possible values: {coord_types}"
    
    if i_row == 'all':
        df_pose = df_pose.iloc[:,:]
    else:
         df_pose = df_pose.iloc[i_row:i_row+1,:]
            
    axis_names = ['x', 'y', 'z']
    x_y_zs = []


    if coord_type == 'local':
        coord_str = 'pose'
    elif coord_type == 'global':
        coord_str = 'pose_world'
    else:
        print('Warning! Bad coord_type!')
    
 
    for axis_name in axis_names:
        columns_i = [f"{coord_str}__{point}_{axis_name}" for point in points] 
        x_y_zs += [df_pose[columns_i]]
        
    if to_np:
        for i in range(len(x_y_zs)):
            x_y_zs[i] = x_y_zs[i].to_numpy()
            if x_y_zs[i].shape[0] == 1:
                x_y_zs[i] = x_y_zs[i].reshape(-1)
                        
    xs, yx, zs = x_y_zs
    
    return xs, yx, zs


def get_main_bodypart_tensor(pose_tensor, points_axis=2 ):
    points = list(range(0, 23))
    return pose_tensor.take(points, axis=points_axis)


def get_main_bodypart_coordinates(df_pose, i_row=0, to_np=True, coord_type='local'):
    points = list(range(0, 23))
    return get_pose_points_coordinates(df_pose, points, i_row, to_np=to_np, coord_type=coord_type)


def get_allbody_coordinates(df_pose, i_row=0, to_np=True, coord_type='local'):
    points = list(range(0, 33))
    return get_pose_points_coordinates(df_pose, points, i_row, to_np=to_np, coord_type=coord_type)
 

def get_hands_line_coordinates(df_pose, i_row=0, to_np=True, coord_type='local'):    
    points = [21, 15, 17, 19, 15, 13, 11, 12, 14, 16, 20, 18, 16, 22]
    return get_pose_points_coordinates(df_pose, points, i_row, to_np=to_np, coord_type=coord_type)

   
def get_face_line_coordinates(df_pose, i_row=0, to_np=True, coord_type='local'):
    points = [0, 10, 9, 0, 8, 6, 5, 4, 0, 7, 3, 2, 1, 0]
    return get_pose_points_coordinates(df_pose, points, i_row, to_np=to_np, coord_type=coord_type)


def get_boxbody_line_coordinates(df_pose, i_row=0, to_np=True, coord_type='local'):
    points = [11, 12, 24, 23, 11]
    return get_pose_points_coordinates(df_pose, points, i_row, to_np=to_np, coord_type=coord_type)


def get_legs_line_coordinates(df_pose, i_row=0, to_np=True, coord_type='local'):
    points = [28, 32, 30, 28, 26, 24, 23, 25, 27, 29, 31, 27]
    return get_pose_points_coordinates(df_pose, points, i_row, to_np=to_np, coord_type=coord_type)
    
    
"""
Tensors: np.array[xs, yx, zs]
"""

def get_pose_points_tensor(df_pose, points, i_row='all', xyz_axis=0,  coord_type='local'):
    xs, yx, zs = get_pose_points_coordinates(df_pose, points, i_row=i_row,  coord_type=coord_type,  to_np=True)
    tensor = np.stack([xs, yx, zs], axis=xyz_axis)
    return tensor

def get_pose_tensor(df_pose, i_row='all', xyz_axis=0, coord_type='local'):
    points = list(range(0, 33))
    return get_pose_points_tensor(df_pose, points, i_row, xyz_axis=xyz_axis, coord_type=coord_type)
    


# 3) transform coordinates (scale, rotate)

"""
    Transform local coordinates in poxels into global coordinates in meters using mediapipe global coordinate
"""
# Using linear regression  

#functio 
def loc_to_glob_predict(loc_t, glob_t, n=-1):

    if n == -1:
        n == glob_t.shape[-1]

    glob_t_predicted = glob_t.copy()

    for i in range(glob_t_predicted.shape[1]):
        X1= loc_t[:, i, 0:n].T
        Y1 = glob_t[:, i, 0:n]

        lr_list = [LinearRegression() for i in range(Y1.shape[0])]

        for ii, lr in enumerate(lr_list):
            lr.fit(X1, Y1[ii])

        for j in range(glob_t_predicted.shape[0]):
            glob_t_predicted[j, i, :] = lr_list[j].predict(loc_t[:,i, :].T)

    return glob_t_predicted


# 4) data loading
"""
Loading data
"""
def preproc_sample(df, sep_in='.', sep_out='__', sep_coord='_'):
    # df = pd.read_csv(load_path)
    # print(list(df_row.columns))

    new_columns = []
    for column in df.columns:
        
        column_splitted = column.split(sep_in)
        # print(column_splitted)
        if column_splitted[0] == 'pose':
            enum = POSE_ENUM
            # print(enum[column_splitted[1].upper()])
            column_splitted[1] = str(enum[column_splitted[1].upper()])
            column = column_splitted[0] + sep_out + column_splitted[1] + sep_coord + column_splitted[2]
        elif column_splitted[0] == 'world_pose':
            enum = POSE_ENUM
            # print(enum[column_splitted[1].upper()])
            column_splitted[1] = str(enum[column_splitted[1].upper()])
            column = 'pose_world' + sep_out + column_splitted[1] + sep_coord + column_splitted[2]
        elif column_splitted[0] in ['right_hand', 'left_hand']:
            enum = HAND_ENUM
            column_splitted[1] = str(enum[column_splitted[1].upper()])
            column = column_splitted[0] + sep_out + column_splitted[1] + sep_coord + column_splitted[2]



        new_columns += [column]

    df.columns = new_columns
    return df



def load_sample(load_path, sep_in='.', sep_out='__', sep_coord='_'):
    df = pd.read_csv(load_path)
    return preproc_sample(df, sep_in, sep_out, sep_coord)








# Coordinate transformation   

def main_pose_data_tranform(XYZs, transfor_type='local'):
    if transfor_type == 'local':
        XYZs[0] = XYZs[0]
        XYZs[1] = XYZs[1]
        XYZs[2] = XYZs[2]

    # y <-> z
    XYZs[:,:,:] = XYZs[[0, 2, 1], :, :]
    # z -> -z
    XYZs[2] *= -1
    # z -> min(z) = 0 
    XYZs[2] = XYZs[2] - XYZs[2].min(axis=1, keepdims=True)
    return XYZs


def main_hand_data_tranform(XYZs, pose_XYZ=None, hand_type='left',  transfor_type='local'):

    if pose_XYZ is not None:
        wrist_name = (hand_type + '_WRIST').upper()
        # if transfor_type == 'local':
        dz = pose_XYZ[2, :, POSE_ENUM[wrist_name]]
        dz = np.stack([dz]*XYZs.shape[2], axis=1)
        XYZs[2, :, :] += dz
    else:
        XYZs = XYZs - XYZs[:, :, 0:1]
        XYZs = np.nan_to_num(XYZs)


    

    # # y <-> z
    # XYZs[:,:,:] = XYZs[[0, 2, 1], :, :]
    # # z -> -z
    # XYZs[2] *= -1
    # # z -> min(z) = 0 
    # XYZs[2] = XYZs[2] - XYZs[2].min(axis=1, keepdims=True)
    return XYZs






def main_hands_visual_tranform(XYZs, hand_type='left',  transfor_type='local'):
    # wrist at the origin

    XYZs = XYZs - XYZs[:, :, 0:1]
    XYZs = np.nan_to_num(XYZs)
    # Xs = np.array([[], [], []])

    # det = np.linalg.det()

    # if pose_XYZ is not None:
    #     wrist_name = (hand_type + '_WRIST').upper()
    #     # if transfor_type == 'local':
    #     dz = pose_XYZ[2, :, POSE_ENUM[wrist_name]]
    #     dz = np.stack([dz]*XYZs.shape[2], axis=1)
    #     XYZs[2, :, :] += dz


    # # y <-> z
    # XYZs[:,:,:] = XYZs[[0, 2, 1], :, :]
    # # z -> -z
    # XYZs[2] *= -1
    # # z -> min(z) = 0 
    # XYZs[2] = XYZs[2] - XYZs[2].min(axis=1, keepdims=True)
    return XYZs










# Plot data and visualozation
    
def gen_pose_plot_data(XYZs, row_idx=0, common_color=None, line_width=1, point_size=2):
    lines_names = ['Hands', 'Face', 'Legs', 'Boxbody']
    points_name = 'Points'

    xyzs = XYZs[:, :, POSE_POINTS]
    xyzs_h = XYZs[:, :, HANDS_LINE_POINTS]
    xyzs_f = XYZs[:, :, FACE_LINE_POINTS]
    xyzs_b = XYZs[:, :, BOXBODY_LINE_POINTS]
    xyzs_l = XYZs[:, :, LEGS_LINE_POINTS]

    
    i = row_idx
    xs, ys, zs = xyzs[:,i,:]

    points_list = [xyzs_h[:, i, :], xyzs_f[: ,i, :], xyzs_l[:, i, :], xyzs_b[:, i, :]]


    point_color = 'blue'
    line_color = None
    if common_color is not None:
        point_color = common_color
        line_color = common_color

    
    line_marker = {'width': line_width, 'color': line_color}
    marker_marker = dict(size=point_size, color=point_color, colorscale='Viridis')

    plot_data =[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=marker_marker, 
                          name=points_name)]
    
    plot_data_lines = [go.Scatter3d(x=points[0], y=points[1], z=points[2],  mode='lines', name=name, line=line_marker)
                       for name, points in zip(lines_names, points_list) ]
    plot_data += plot_data_lines
    return plot_data


def gen_hand_plot_data(XYZs, row_idx=0, common_color=None, line_width=1, point_size=2):
    lines_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Palmbox']
    points_name = 'Hand Points'


    xyzs = XYZs[:, :, HAND_POINTS]
    i = row_idx
    xs, ys, zs = xyzs[:,i,:]


    PALM_LINES = [THUMB_LINE_POINTS, INDEX_LINE_POINTS, MIDDLE_LINE_POINTS, RING_LINE_POINTS, PINKY_LINE_POINTS, PALMBOX_LINE_POINTS]
    xyz_palm_lines = [XYZs[:, :, finger_line_points] for finger_line_points in PALM_LINES]

    points_list = [xyzs_[:, i, :] for xyzs_ in xyz_palm_lines]

    point_color = 'blue'
    line_color = None
    if common_color is not None:
        point_color = common_color
        line_color = common_color

    
    line_marker = {'width': line_width, 'color': line_color}
    marker_marker = dict(size=point_size, color=point_color, colorscale='Viridis')

    plot_data =[go.Scatter3d(x=xs, y=ys, z=zs, mode='markers', marker=marker_marker, 
                          name=points_name)]
    plot_data_lines = [go.Scatter3d(x=points[0], y=points[1], z=points[2], mode='lines', name=name, line=line_marker)  
                       for name, points in zip(lines_names, points_list) ]
    plot_data += plot_data_lines
    return plot_data



    




def make_frames(XYZs, range_=-1, gen_plot_data='pose', common_color=None, point_size=2, line_width=1):

    str_gen_plot_data = ['pose'] + ['left', 'left_hand', 'l_hand', 'right', 'right_hand', 'r_hand']

    if isinstance(gen_plot_data, str):
        if gen_plot_data.lower() == 'pose':
            gen_plot_data = gen_pose_plot_data
        elif gen_plot_data.lower() in ['left', 'left_hand', 'l_hand', 'right', 'right_hand', 'r_hand',]:
            gen_plot_data = gen_hand_plot_data
        else:
            print(f"Error, get_plot_data={gen_plot_data} is not defined, possible_values: {str_gen_plot_data}")

    
    row_axis = 1

    if range_ == -1:
        range_ = [0, XYZs.shape[row_axis], 1]
    assert isinstance(range_, (tuple, list)), f"Error, range_ should be list or tuple, or -1, given:{range_} , type: {type(range_)}"

    frames = []
    # for i in range(*range_[0:3]):
    for i in range(0,XYZs.shape[row_axis], 1):
        plot_data = gen_plot_data(XYZs, i, common_color, point_size=point_size, line_width=line_width)
        frames += [go.Frame(data=plot_data, name=str(i))]
    return frames

    
def init_sliders_dict():
    sliders_dict =  {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 20},
                            "prefix": "Frame:",
                            "visible": True,
                            "xanchor": "right"
                        },
                        "transition": {"duration": 50, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": []
    }
    return sliders_dict



def create_slider_for_frames(frames: list, sliders_dict=None):
    if sliders_dict is None:
        sliders_dict = init_sliders_dict()
    
    for frame in frames:
        n_frame = frame['name']
        slider_step = {"args": [
                                [n_frame],
                                {"frame": {"duration": 50, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 50}},
                                ],
                        "label": n_frame,
                        "method": "animate",
                        }

        sliders_dict["steps"].append(slider_step)
    return sliders_dict


def init_updatemenus(duration='def'):
    if duration == 'def':
        duration = int(1/30*10)
        
    updatemenus=[{"buttons": [
                            {
                                "args": [None, {"frame": {"duration": duration, "redraw": True},
                                                "fromcurrent": True, "transition": {"duration": duration,
                                                                                    "easing": "quadratic-in-out"}}],
                                "label": "Play",
                                "method": "animate"
                            },
                            {
                                "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                                "mode": "immediate",
                                                "transition": {"duration": 0}}],
                                "label": "Pause",
                                "method": "animate"
                            }
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 87},
                        "showactive": False,
                        "type": "buttons",
                        "x": 0.1,
                        "xanchor": "right",
                        "y": 0,
                        "yanchor": "top"
                    }]
                    
    return updatemenus





def create_slider_animation(frames, sliders_dict, duration='def'):
    if duration == 'def':
        duration = int(1/30*10)

    fig = go.Figure(
                    data=frames[0]['data'],
                    layout=go.Layout(
                        scene = dict(xaxis = dict(nticks=5, range=[-1, 1], autorange=False),
                                    yaxis = dict(nticks=14, range=[-1, 1], autorange=False),
                                    zaxis = dict(nticks=6, range=[0, 2], autorange=False),
                                    aspectmode  ='cube',
                        ),
                        width=700,
                        height=500,
                        # margin=dict(r=20, l=10, b=10, t=10),
                        xaxis=dict(range=[0, 5], autorange=False),
                        yaxis=dict(range=[0, 5], autorange=False),
                        title="Start Title",
                        updatemenus=init_updatemenus(duration),
                        hovermode="closest",
                        
                    sliders= [sliders_dict],      
                    ),
                    frames=frames[:]
    )
    fig['frames'][-1]['layout']=go.Layout(title_text="End Title")
        

    return fig












#class
class CoordinateTransformer:
    def __init__(self, normalize=True, n=-1):
        self.n = n
        self.normalize = normalize

    def fit(self, loc_t, glob_t, n_train=-1):
        if self.n == -1:
            self.n == glob_t.shape[-1]

        if n_train == -1:
            n_train = glob_t.shape[1]


        self.lr_list = [LinearRegression(normalize=self.normalize) for i in range(glob_t.shape[0])]
        # X1_list, Y1_list = [], []
        # for i in range(n_train):
        i = n_train
        X1= loc_t[:, i, 0:self.n].T
        Y1 = glob_t[:, i, 0:self.n]
            # X1_list += [X1]
            # Y1_list += [Y1]

        # X1 = np.concatenate(X1_list, axis=0)
        # Y1 = np.concatenate(Y1_list, axis=1)

        for i, lr in enumerate(self.lr_list):
            lr.fit(X1, Y1[i])

        pass


    def predict(self, loc_t):
        glob_t_predicted = loc_t.copy()

        for i in range(glob_t_predicted.shape[1]):
            for j in range(glob_t_predicted.shape[0]):
                glob_t_predicted[j, i, :] = self.lr_list[j].predict(loc_t[:,i, :].T)

        return glob_t_predicted



