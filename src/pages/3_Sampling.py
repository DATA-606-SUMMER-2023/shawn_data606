import streamlit as st
import pandas as pd
import numpy as np
import random

st.markdown('# Sampling')

feature_names = list(st.session_state['feature_df'].columns)
df = pd.DataFrame({
    'Feature': [f for f in feature_names],
    'Include': False,
    'Cuts': 2
})
df_edit = st.data_editor(df)

def sample():

    feature_df = st.session_state['feature_df']
    selected_feature_df = df_edit[df_edit['Include']]

    feature_df1 = feature_df[feature_df['Animation Length'] > 1000]
    feature_df1 = feature_df1[list(selected_feature_df['Feature'])].dropna()

    cuts = {
        f + ' Cut': pd.qcut(feature_df1[f], c, labels=[f + ' ' + str(i+1) for i in range(c)])
        for f, c in zip(selected_feature_df['Feature'], selected_feature_df['Cuts'])
    }

    d1 = feature_df1.assign(**cuts)
    # d1['bin'] = pd.Categorical(d1.filter(regex='Cut').apply(tuple, 1))

    # print(list(d1.filter(regex='Cut').columns))
    # d1['bin'].value_counts()
    # d1.groupby()

display_sample_menu = st.button('Get Sample Set')
get_samples = False

if 'display_sample_menu' in st.session_state:
    if True not in list(df_edit['Include']):
        st.session_state['display_sample_menu'] = False

if display_sample_menu or st.session_state.get('display_sample_menu'):

    st.session_state['display_sample_menu'] = True

    feature_df = st.session_state['feature_df']
    selected_feature_df = df_edit[df_edit['Include']]

    feature_df1 = feature_df[feature_df['Animation Length'] > 1000]
    feature_df1 = feature_df1[list(selected_feature_df['Feature'])].dropna()

    cuts = {
        f + ' Cut': pd.qcut(feature_df1[f], c, labels=[f + ' ' + str(i+1) for i in range(c)])
        for f, c in zip(selected_feature_df['Feature'], selected_feature_df['Cuts'])
    }

    d1 = feature_df1.assign(**cuts)
    bin_counts = d1.groupby([col for col in d1 if 'Cut' in col]).size()

    st.write(bin_counts)

    bin_size = st.number_input('Samples per bin', 1)
    frame_size = st.number_input('Frames per motion', 100)
    train_test_split = st.slider('Train/Test Split', 0.0, 1.0, 0.8)

    train_set_name = st.text_input('Train Set Name') if train_test_split != 0 else None
    test_set_name = st.text_input('Test Set Name') if train_test_split != 1 else None

    # sample_set_name = st.text_input('Sample Set Name')

    get_samples = st.button('Get Samples')

    if get_samples:

        selected_motions = list(d1.groupby([col for col in d1 if 'Cut' in col]).sample(bin_size).index)

        st.write(f'{len(selected_motions)} sampled motions with {frame_size} sampled frames each included in training set.')
        # selected_motion_paths = [st.session_state['motion_paths'][m + '.pkl'] for m in selected_motions]
        sample_selection = {}

        for motion in selected_motions:

            num_frames = feature_df.loc[motion, 'Animation Length']
            sample_selection[motion] = np.random.choice(num_frames, size=int(frame_size), replace=False)

        st.session_state['sample_selection'] = sample_selection

        pbar1 = st.progress(0.0)
        num_motions = len(st.session_state['sample_selection'].keys())
        sample_set_df = pd.DataFrame()

        for i, (motion, frames) in enumerate(st.session_state['sample_selection'].items()):

            pbar1.progress(i/num_motions, f"Processing motion {motion}")

            file_path = st.session_state['motion_paths'][motion + '.pkl']
            df = pd.read_pickle(file_path)[['position', 'rotation']]
            df = df[[f in frames for f in df.index.get_level_values('frame')]]
            df = df.reset_index()
            df['motion'] = motion

            if sample_set_df.empty:
                sample_set_df = df.copy()
            else:
                sample_set_df = pd.concat([sample_set_df, df])

        motions = list(sample_set_df['motion'].unique())
        train_motions = random.sample(motions, int(len(motions) * train_test_split))

        train_set_df = sample_set_df[sample_set_df['motion'].isin(train_motions)]
        test_set_df = sample_set_df[~sample_set_df['motion'].isin(train_motions)]

        if train_set_name is not None:
            train_set_df.to_pickle(f'data/sample_sets/{train_set_name}')

        if test_set_name is not None:
            test_set_df.to_pickle(f'data/sample_sets/{test_set_name}')

        pbar1.progress(1.0, 'Done!')
        


