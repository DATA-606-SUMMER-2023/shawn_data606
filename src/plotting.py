import os
import regex
import pickle

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

import plotly
import plotly.express as px
import plotly.graph_objects as go

def get_pose_frame(pose_df, lines=True, tracker_size=20, prediction_size=10, joint_size=1, joint_color=None):
    '''
    Provides a frame to use in a plotly figure based on the provided pose_df.

    Arguments:
    pose_df -- Dataframe containing pose information. Must include at least the position, rotation, direction, length, and children of joints.

    Keywords Arguments:
    tracker_size -- default size of markers for trackers
    prediction_size -- default size of markers for predicted joint locations
    joint_size -- default size of markers for regular joints
    '''
    
    use_lines = 'direction' in pose_df.columns and lines
    
    is_tracker = pose_df.index.get_level_values('joint').str.contains('tracker')
    is_pred = pose_df.index.get_level_values('joint').str.contains('pred')
    
    # Position markers
    positions = np.stack(pose_df['position'])
    go_scatter = go.Scatter3d(
        x=positions[:, 2],
        y=positions[:, 0],
        z=positions[:, 1],
        mode='markers',
        marker={
            'size': [tracker_size if j else prediction_size if i else joint_size for i, j in zip(is_tracker, is_pred)],
            'color': joint_color if joint_color is not None else ['green' if j else 'blue' if i else 'black' for i, j in zip(is_tracker, is_pred)]
        },
        hoverinfo='skip',
        hovertemplate='%{text}',
        text=pose_df.index
    )
    
    # Lines
    if use_lines:
        
        pose_mocap_df = pose_df[~is_tracker]
        
        lines = np.array([
            [start, middle, end] for start, middle, end in zip(
                pose_mocap_df['position'], 
                pose_mocap_df['position'] + (pose_mocap_df['direction'] * pose_mocap_df['length'] / 2),
                pose_mocap_df['position'] + (pose_mocap_df['direction'] * pose_mocap_df['length'])
            )
        ])

        go_lines = [
            go.Scatter3d(
                x=line[:, 2], 
                y=line[:, 0], 
                z=line[:, 1],
                mode='lines',
                line={
                    'color': 'black'
                },
                hoverinfo='skip',
                hovertemplate=[None, name, None],
                # text=name
            ) for name, line in zip(pose_mocap_df.index, lines)
        ]

    return go.Frame(data=[go_scatter] + go_lines if use_lines else go_scatter)