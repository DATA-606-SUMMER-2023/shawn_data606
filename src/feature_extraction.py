import os
import re

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from tqdm.auto import tqdm

# My modules
import parsing
import posing
import plotting

def get_torso_verticality(animation_df):
    torso_vert = animation_df.loc[:, 'lowerneck', :]['position'] - animation_df.loc[:, 'lowerback', :]['position']
    torso_vert /= torso_vert.apply(np.linalg.norm)
    torso_vert = torso_vert.apply(lambda x: np.dot(np.array([0, 1, 0]), x))
    return torso_vert

def get_diff(animation_df):

    joints = animation_df.loc[0].index
    final_df = pd.DataFrame()
    
    for j in joints:
        j_df = animation_df.loc[(slice(None), j), :]
        new_df = pd.DataFrame(index = j_df.index)
        new_df['velocity'] = j_df['position'].diff() * 120.0
        new_df['acceleration'] = new_df['velocity'].diff() * 120.0
        new_df['jerk'] = new_df['acceleration'].diff() * 120.0

        final_df = pd.concat([final_df, new_df])

    return final_df.reindex(animation_df.index)

def root_instability(animation_df):
    diff = get_diff(animation_df)
    jerk = diff['jerk'].apply(np.linalg.norm)
    return jerk.loc[:, 'root']

def localize_positions_to_root(animation_df):
    df = animation_df.copy()
    num_joints = len(df.index.get_level_values('joint').unique())
    df_roots = df.loc[:, 'root', :]['position'].repeat(num_joints)
    df['position'] = list(df['position'].reset_index(drop=True) - df_roots.reset_index(drop=True))
    return df

def energy(animation_df, joints=['root']):
    local_df = localize_positions_to_root(animation_df).loc[:, joints, :]
    diff_df = get_diff(local_df).fillna(0)
    return diff_df['velocity'].apply(np.linalg.norm).groupby('frame').sum() / 120.0

def extract_features(animation_df):
    return pd.DataFrame({
        'Torso Verticality Mean': [get_torso_verticality(animation_df).mean()],
        'Torso Verticality Std.': [get_torso_verticality(animation_df).std()],
        'Instability': [root_instability(animation_df).mean()],
        'Hand Energy': [energy(animation_df, joints=['lwrist', 'rwrist']).mean()],
        'Foot Energy': [energy(animation_df, joints=['lfoot', 'rfoot']).mean()],
        'Head Energy': [energy(animation_df, joints=['head']).mean()]
    })