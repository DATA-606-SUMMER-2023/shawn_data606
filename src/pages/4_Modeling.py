import streamlit as st
import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.linear_model import LinearRegression

import modeling

st.markdown('# Modeling')

if 'joints' not in st.session_state:
    st.session_state['joints'] = pd.read_pickle(next(iter(st.session_state['motion_paths'].values()))).index.get_level_values('joint').unique()

if 'last_train' not in st.session_state:
    st.session_state['last_train'] = ''

if 'last_test' not in st.session_state:
    st.session_state['last_test'] = ''

joints = st.session_state['joints']

input_col, output_col = st.columns(2)

with input_col:

    # Input joints
    st.write('Inputs')
    default_input_joints = ['lhand', 'rhand', 'head']
    input_joint_select = st.data_editor(pd.DataFrame({

        'Joint': joints, 
        'Include Position Vector': [j in default_input_joints for j in joints], 
        'Include Rotation Matrix': [j in default_input_joints for j in joints]

    }), disabled=('Joint',), hide_index=True)

with output_col:

    # Output joints
    st.write('Outputs')
    default_output_joints = ['root', 'thorax', 'lhumerus', 'rhumerus']
    output_joint_select = st.data_editor(pd.DataFrame({

        'Joint': joints, 
        'Include Position Vector': [j in default_output_joints for j in joints], 
        'Include Rotation Matrix': False

    }), disabled=('Joint',), hide_index=True)

input_positions = input_joint_select[input_joint_select['Include Position Vector']]['Joint'].to_list()
input_rotations = input_joint_select[input_joint_select['Include Rotation Matrix']]['Joint'].to_list()

output_positions = output_joint_select[output_joint_select['Include Position Vector']]['Joint'].to_list()
output_rotations = output_joint_select[output_joint_select['Include Rotation Matrix']]['Joint'].to_list()

config_dict = {
    'input_positions': input_positions,
    'input_rotations': input_rotations,
    'output_positions': output_positions,
    'output_rotations': output_rotations
}

input_size = len(input_positions) * 3 + len(input_rotations) * 9
output_size = len(output_positions) * 3 + len(output_rotations) * 9

st.write(f'{input_size} input scalars -> {output_size} output scalars')

sample_set_train_name = st.selectbox('Train Set', os.listdir('data/sample_sets'))
sample_set_test_name = st.selectbox('Test Set', ['None'] + os.listdir('data/sample_sets'))

model_name = st.text_input(f'Model Name')

# st.number_input('MSE Positional Loss Weight', .75)
# st.number_input('MSE Rotational Loss Weight', .25)

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
    input_joints=['lhand', 'rhand', 'head'],
    output_joints=['root', 'thorax', 'lhumerus', 'rhumerus'],
    flatten=True
):
    
    if isinstance(frame_df.index, pd.MultiIndex):
        samples = [
            frame_to_sample(df.droplevel(0), input_joints, output_joints, flatten) 
            for i, df in frame_df.groupby('frame', group_keys=False)
        ]
        return [i[0] for i in samples], [i[1] for i in samples]
        
    else:
        X, y = np.stack(frame_df.loc[input_joints]['position']), np.stack(frame_df.loc[output_joints]['position'])
        return (X, y) if not flatten else (X.flatten(), y.flatten())

def sample_to_frame(
    X,
    y,
    input_joints=['lhand', 'rhand', 'head'],
    output_joints=['root', 'thorax', 'lhumerus', 'rhumerus']
):
    
    if isinstance(X, list):
        return pd.concat([
            sample_to_frame(i, j, input_joints, output_joints)
            for i, j in zip(X, y)
        ], keys=range(len(X)))
    
    return pd.DataFrame(
        {'position': list(np.concatenate([X.reshape(3, 3), y.reshape(4, 3)]))}, 
        index=input_joints + output_joints
    )

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

# Train model

model_type_dict = {
    'Linear': modeling.LinearModel,
    'Quadratic': modeling.QuadraticModel,
    'Random Forest': modeling.RandomForestModel,
    'Support Vector Machine': modeling.SVMModel
}

model_type = st.selectbox('Model Type', model_type_dict.keys())

if st.button('Train'):

    def sample_set_to_X_y(sample_set_df):

        X = []
        y = []

        pbar2 = st.progress(0.0)
        num_frames = len(sample_set_df.groupby(['motion', 'frame']).size())

        for j, ((motion, frame), frame_df) in enumerate(sample_set_df.groupby(['motion', 'frame'])):
            
            pbar2.progress(j/num_frames, f'Processing motion {motion} frame {frame}')

            input_df = frame_df[frame_df['joint'].isin(input_positions)].set_index('joint', drop=True).reindex(input_positions)
            output_df = frame_df[frame_df['joint'].isin(output_positions)].set_index('joint', drop=True).reindex(output_positions)

            X_i_pos = np.concatenate(input_df['position'])
            X_i_rot = np.concatenate(input_df['rotation'].apply(lambda x: x.flatten()))
            X_i = np.concatenate([X_i_pos, X_i_rot])

            y_i = np.concatenate(output_df['position'])

            X.append(X_i)
            y.append(y_i)

        pbar2.progress(1.0, 'Done!')

        return X, y
    
    # Train

    if st.session_state['last_train'] != sample_set_train_name:
        train_set_df = pd.read_pickle(f'data/sample_sets/{sample_set_train_name}')
        X, y = sample_set_to_X_y(train_set_df)

    else:
        X, y = st.session_state['last_train_X'], st.session_state['last_train_y']

    model = model_type_dict[model_type]().fit(X, y)

    # Test

    losses = None

    if sample_set_train_name != 'None':

        if st.session_state['last_test'] != sample_set_test_name:

            test_set_df = pd.read_pickle(f'data/sample_sets/{sample_set_test_name}')
            X_test, y_test = sample_set_to_X_y(test_set_df)

        else:
            X_test, y_test = st.session_state['last_test_X'], st.session_state['last_test_y']

        y_pred = model.predict(X_test)

        losses = {}
        losses['all'] = mean_squared_error(y_test, model.predict(X_test))

        for i, o in enumerate(output_positions):
            losses[o] = mean_squared_error([j[i*3:i*3+3] for j in y_test], [j[i*3:i*3+3] for j in y_pred])

        st.write(pd.DataFrame({
            'Joint': losses.keys(),
            'MSE Loss': losses.values()
        }).set_index('Joint'))
        
    # st.write(f'Linear model R score: {linear_model.score(X, y)}')

    st.session_state['model'] = model

    with open(f'models/{model_name}', 'wb') as f:
        pickle.dump({
            'model': st.session_state['model'], 
            'config': config_dict,
            'metrics': tuple(
                (a, b) for a, b in losses.items()
            )
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    st.session_state['last_train'] = sample_set_train_name
    st.session_state['last_test'] = sample_set_test_name
    st.session_state['last_train_X'] = X
    st.session_state['last_train_y'] = y
    st.session_state['last_test_X'] = X
    st.session_state['last_test_y'] = y