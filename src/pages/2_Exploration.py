import streamlit as st
import pandas as pd
import os
import sys

import plotly.express as px

# import sys
# sys.path.append('..')

# My modules
import feature_extraction

st.markdown('# Exploration')

feature_path = os.path.join(st.session_state['data_path'], 'features.csv')

if 'feature_df' in st.session_state:

    with st.expander('Dataframe'):

        st.write(st.session_state['feature_df'])
        
elif os.path.isfile(feature_path):

    st.session_state['feature_df'] = pd.read_csv(feature_path, index_col=0)

    with st.expander('Dataframe'):

        st.write(st.session_state['feature_df'])

else:

    data_path = st.session_state['data_path']

    if data_path is None:
        st.error('No data found. See Data section.')

    pbar = st.progress(0.0, f"Reading {next(iter(st.session_state['motion_paths']))}")
    num_motions = len(st.session_state['motion_paths'].keys())

    feature_df = pd.DataFrame()

    for p, motion_path in enumerate(st.session_state['motion_paths'].values()):
        motion_df = pd.read_pickle(motion_path)
        features = feature_extraction.extract_features(motion_df)
        features.index = [os.path.basename(motion_path).replace('.pkl', '')]
        feature_df = pd.concat([feature_df, features])
        pbar.progress(p/num_motions, f"Reading {motion_path}")

    pbar.progress(1.0, 'Done!')

    with st.expander('Dataframe'):

        st.session_state['feature_df'] = feature_df.sort_index()
        st.write(feature_df)

with st.expander('Correlation Matrix'):

    st.write(px.imshow(st.session_state['feature_df'].corr()))

feature_names = list(st.session_state['feature_df'].columns)
feature_tabs = st.tabs(feature_names)

for i, feature in enumerate(feature_names):

    with feature_tabs[i]:

        selection = st.selectbox('Visualization', ('Histogram', 'Scatterplot'), key=feature)

        if selection == 'Histogram':
            st.write(px.histogram(st.session_state['feature_df'], x=feature))
        else:
            st.write(px.scatter(st.session_state['feature_df'], y=feature))