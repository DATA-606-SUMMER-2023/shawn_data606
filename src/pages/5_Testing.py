import streamlit as st

import os
import re
import pickle
import math

import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression

import plotly.graph_objects as go
import plotly.express as px
from tqdm.auto import tqdm

# My modules
import parsing
import posing
import plotting
import feature_extraction
import modeling

st.markdown('# Testing')

def get_frames(motion, frames=None, subject_path='../data/subjects/'):
    subject, trial = motion.split('_')
    pkl = motion + '.pkl'

    anim_df = pd.read_pickle(os.path.join(subject_path, subject, pkl))
    if isinstance(frames, list):
        anim_df = anim_df.loc[frames, :]
    elif frames is not None:
        anim_df = anim_df.loc[frames]
    return anim_df

def frame_to_sample(
    frame_df, 
    input_positions=['lhand', 'rhand', 'head'],
    input_rotations=['lhand', 'rhand', 'head'],
    output_positions=['root', 'thorax', 'lhumerus', 'rhumerus'],
    output_rotations=['root', 'thorax', 'lhumerus', 'rhumerus'],
    flatten=True
):
    
    if isinstance(frame_df.index, pd.MultiIndex):
        samples = [
            frame_to_sample(df.droplevel(0), input_positions, input_rotations, output_positions, output_rotations, flatten) 
            for i, df in frame_df.groupby('frame', group_keys=False)
        ]
        return [i[0] for i in samples], [i[1] for i in samples]
        
    else:

        X, y = np.stack(frame_df.loc[input_positions]['position']), np.stack(frame_df.loc[output_positions]['position'])

        if input_rotations:
            X = np.vstack([np.array(X), np.array(list(frame_df.loc[input_rotations]['rotation'])).reshape(-1, 3)])

        if output_rotations:
            y = np.vstack([np.array(y), np.array(list(frame_df.loc[output_rotations]['rotation'])).reshape(-1, 3)])

        return (X, y) if not flatten else (X.flatten(), y.flatten())

def sample_to_frame(
    X,
    y,
    input_positions=['lhand', 'rhand', 'head'],
    input_rotations=['lhand', 'rhand', 'head'],
    output_positions=['root', 'thorax', 'lhumerus', 'rhumerus'],
    output_rotations=['root', 'thorax', 'lhumerus', 'rhumerus'],
):
    
    if isinstance(X, list):
        return pd.concat([
            sample_to_frame(i, j, input_positions, input_rotations, output_positions, output_rotations)
            for i, j in zip(X, y)
        ], keys=range(len(X)))
    
    return pd.DataFrame(
        {'position': list(np.concatenate([X.reshape(-1, 3)[:len(input_positions)], y.reshape(-1, 3)[:len(output_positions)]]))}, 
        index=input_positions + output_positions
    )
    
    # if use_rotation:

    #     return pd.DataFrame(
    #         {'position': list(np.concatenate([X.reshape(-1, 3)[:3], y.reshape(4, 3)]))}, 
    #         index=input_joints + output_joints
    #     )

    # else:
    #     return pd.DataFrame(
    #         {'position': list(np.concatenate([X.reshape(3, 3), y.reshape(4, 3)]))}, 
    #         index=input_joints + output_joints
    #     )

def plot_pose(pose_df, prediction_df=None):
    '''
    Creates a plot showing a pose_df, which should wither be a rest pose or a pose with a motion applied.

    Arguments:
    pose_df - Dataframe containing pose information. Must include at least the position, rotation, direction, length, and children of joints.
    '''
    # pose = pose_df
    # if prediction:
    #     pose.index = [i + '_pred' for i in pose_df.index]
    #     pose.index.name = 'joint'

    frame = plotting.get_pose_frame(pose_df, joint_size=5, joint_color=None)

    data = frame.data if prediction_df is None else frame.data + plotting.get_pose_frame(prediction_df, joint_size=10, joint_color='green').data

    fig = go.Figure(
        data=data
    )

    root_position = pose_df.loc['root', 'position']

    fig.update_layout(
        scene={
            'aspectmode': 'cube',
            'xaxis': {'range': (root_position[2]-20, root_position[2]+20)},
            'yaxis': {'range': (root_position[0]-20, root_position[0]+20)},
            'zaxis': {'range': (root_position[1]-20, root_position[1]+20)},
            'xaxis_title': 'z',
            'yaxis_title': 'x',
            'zaxis_title': 'y'
        },
    )

    return fig

model_selection = st.selectbox('Model', os.listdir('models'))

with open(f'models/{model_selection}', 'rb') as f:
    files = pickle.load(f)
    st.session_state['model'] = files['model']
    st.session_state['model_config'] = files['config']
    st.session_state['model_loss'] = files['metrics']

motion_select = st.selectbox('Motion', st.session_state['feature_df'].index)

st.write(pd.DataFrame({
    'Joint': [i[0] for i in st.session_state['model_loss']],
    'MSE Loss': [i[1] for i in st.session_state['model_loss']]
}).set_index('Joint'))

frame_select = st.slider('Frame', 0, st.session_state['feature_df'].loc[motion_select, 'Animation Length'])

# Predict

frame_df = get_frames(motion_select, frames=frame_select, subject_path=st.session_state['data_path'])
X, y = frame_to_sample(frame_df, **st.session_state['model_config'])
prediction = sample_to_frame(X, st.session_state['model'](X), **st.session_state['model_config'])
prediction.index.name = 'joint'

error = (frame_df[['position']] - prediction).loc[st.session_state['model_config']['output_positions']]
error_list = [np.linalg.norm(pos['position']) for idx, pos in error.iterrows()]
error['error'] = error_list
st.write(error)

# Plot

st.write(plot_pose(frame_df, prediction))