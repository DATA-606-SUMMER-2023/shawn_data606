import os
import regex
import pickle

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

def apply_pose(rest_df, motion_col):
    """
    Takes one columns from a motion dataframe and applies the motions to a rest pose dataframe, creating a new pose.
    
    Arguments:
    rest_df -- the dataframe created from an asf file, outputted from subject_to_dfs in parsing
    motion_col -- one column of a motion dataframe, representing a keyframe in an animation, also from subject_to_dfs
    """

    dof_defaults = {'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0}
    transforms = [
        dof_defaults | dict(zip(dof, motion_col[name])) if name in motion_col.index else dof_defaults
        for name, dof in rest_df['dof'].items()
    ]
    transform_df = pd.DataFrame(transforms, index=rest_df.index)

    pose_df = rest_df.copy()

    def recursive_transform(bone_name, t_offset, r_offset):

        transform = transform_df.loc[bone_name]

        rest_rotation = rest_df.at[bone_name, 'rotation']
        rotation = Rotation.from_euler('xyz', (transform['rx'], transform['ry'], transform['rz']), degrees=True).as_matrix()
        axis = Rotation.from_euler('xyz', pose_df.at[bone_name, 'axis'], degrees=True).as_matrix()
        axis_inverse = np.transpose(axis)
        translation = np.array((transform['tx'], transform['ty'], transform['tz']))

        pose_df.at[bone_name, 'position'] = translation + t_offset
        pose_df.at[bone_name, 'rotation'] = np.linalg.multi_dot([r_offset, axis, rotation, axis_inverse, rest_rotation])
        pose_df.at[bone_name, 'direction'] = np.matmul(pose_df.at[bone_name, 'rotation'], pose_df.at[bone_name, 'direction'].T).T
        
        for child_name in pose_df.at[bone_name, 'children']:

            child_local_position = rest_df['position'][child_name] - rest_df['position'][bone_name]

            new_t_offset = (pose_df.at[bone_name, 'position'] + np.matmul(pose_df.at[bone_name, 'rotation'], child_local_position)).copy()
            new_r_offset = pose_df.at[bone_name, 'rotation'].copy()
            
            recursive_transform(child_name, new_t_offset, new_r_offset)

    recursive_transform('root', pose_df['position']['root'], pose_df['rotation']['root'])

    return pose_df

# def add_tracker(
#     pose_df,
#     tracker_name,
#     parent_name,
#     bone_direction=None,
#     bend_direction=None,
#     global_position=None,
#     local_position=np.array([0., 0., 0.]),
#     length_along_bone_norm=0
# ):
#     """
#     Adds virtual trackers as joints to a pose dataframe, attaching them to a specified joint.
    
#     Arguments:
#     pose_df -- Either a rest dataframe of a pose dataframe outputted from apply_pose()
#     parent_name -- Name of a joint existing in pose_df. The tracker will become a child of this joint.
    
#     Keyword arguments:
#     bone_direction -- The direction the tracker's bone extends in. Used to determine rotation. If None, uses the parent's direction.
#     bend_direction -- The direction the tip of the tracker's bone travels when the joint begins to bend. Used to determine rotation. If None, the rotation of the tracker is the rotation of the parent.
#     local_position -- Position of the tracker relative to its parent.
#     length_along_bone_norm -- Offsets the tracker along the bone of the parent joint. 0 means the tracker will appear at the base of the bone, while 1 means the tracker will appear at the tip.
#     """
    
#     parent = pose_df.loc[parent_name]
#     parent['children'].append(tracker_name)

#     if global_position is None:
#         global_position = parent['position']

#     tracker_dict = {
#         'position': global_position + np.matmul(parent['rotation'], local_position) + (parent['direction'] * parent['length'] * length_along_bone_norm),
#         'rotation': parent['rotation'],
#         'direction': bone_direction if bone_direction is not None else parent['direction'],
#         'length': 0.,
#         'axis': np.array([0., 0., 0.]),
#         'dof': [],
#         'limits': [],
#         'children': []
#     }

#     if bend_direction is not None:
#         e2 = tracker_dict['direction']
#         e2 /= np.linalg.norm(e2)
#         e3 = bend_direction
#         e3 -= e2 * np.dot(e3, e2)
#         e3 /= np.linalg.norm(e3)
#         e1 = np.cross(e2, e3)
#         tracker_dict['rotation'] = np.stack([e1, e2, e3]).T

#     pose_df.loc[tracker_name] = tracker_dict

# def create_mocap_df(rest_df, motion_dfs, position_rotation_only=False, trackers_only=False, plotly_info=False, use_tqdm=False):
#     """
#     Creates a dataframe consisting of all the motion dataframes applied to a rest pose within a subject.
    
#     Arguments:
#     rest_df -- the dataframe of a rest pose
#     motion_dfs -- list of motion dataframes. Each one should represent one animation.
    
#     Keyword Arguments:
#     position_rotation_only -- provide only the position and rotation of each joint, not calculating velocity and acceleration.
#     trackers_only -- only keep joints with "tracker" in the name.
#     plotly_info -- unused now, but would create marker info for plotly plotting.
#     use_tqdm -- Show a progress barfor creating a mocap_df.
#     """

#     if plotly_info:
#         rest_df.loc[:, 'marker_color'] = 'black'
#         rest_df.loc[:, 'marker_size'] = 10
#         rest_df.loc[:, 'draw_rotation'] = False

#     add_tracker(rest_df, 'head_tracker', 'head', bend_direction=np.array([0., 0., 1.]))

#     add_tracker(rest_df, 'hip_tracker', 'root', bone_direction=np.array([0., 1., 0.]), bend_direction=np.array([0., 0., 1.]))
#     add_tracker(rest_df, 'chest_tracker', 'thorax', bend_direction=np.array([0., 0., 1.]))

#     add_tracker(rest_df, 'larm_tracker', 'lhumerus', bend_direction=np.array([0., 0., 1.]))
#     add_tracker(rest_df, 'rarm_tracker', 'rhumerus', bend_direction=np.array([0., 0., 1.]))

#     add_tracker(rest_df, 'lhand_tracker', 'lhand', bend_direction=np.array([0., 0., 1.]))
#     add_tracker(rest_df, 'rhand_tracker', 'rhand', bend_direction=np.array([0., 0., 1.]))

#     add_tracker(rest_df, 'lleg_tracker', 'lfemur', bend_direction=np.array([0., 0., 1.]))
#     add_tracker(rest_df, 'rleg_tracker', 'rfemur', bend_direction=np.array([0., 0., 1.]))

#     add_tracker(rest_df, 'lfoot_tracker', 'lfoot', bend_direction=np.array([0., 1., 0.]))
#     add_tracker(rest_df, 'rfoot_tracker', 'rfoot', bend_direction=np.array([0., 1., 0.]))

#     if plotly_info:
#         rest_df['marker_color'] = rest_df['marker_color'].fillna('green')
#         rest_df['marker_size'] = rest_df['marker_size'].fillna(20)
#         rest_df['draw_rotation'] = rest_df['draw_rotation'].fillna(True)

#     tracker_names = [i for i in rest_df.index if i.endswith('tracker')]

#     mocap_df_list = []
#     j = 0
#     for motion_df in (tqdm(motion_dfs) if use_tqdm else motion_dfs):
    
#         print(f"Applying motion {j} of {len(motion_dfs)}...")
#         j += 1

#         pose_df_list = [rest_df.copy()]

#         for i in motion_df.columns:
            
#             pose_df = apply_pose(rest_df, motion_df[i])

#             if trackers_only:
#                 pose_df = pose_df.loc[tracker_names]

#             if position_rotation_only:
#                 pose_df = pose_df[['position', 'rotation']]

#             pose_df_list.append(pose_df)

#         mocap_df_list.append(pd.concat(pose_df_list, keys=range(len(pose_df_list))))

#     subject_df = pd.concat(mocap_df_list, keys=range(len(mocap_df_list)))
#     subject_df.index = subject_df.index.rename(['trial', 'frame', 'joint'])
    
#     meter_scale = (1.0/0.45)*2.54/100.0
#     subject_df['position'] *= meter_scale
#     subject_df['length'] *= meter_scale
    
#     return subject_df