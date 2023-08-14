import streamlit as st
from streamlit_tree_select import tree_select
import pandas as  pd
import os

st.markdown('# Data')

data_path = os.path.join(os.getcwd(), 'data', 'subjects') if 'data_path' not in st.session_state else st.session_state['data_path']
data_path_input = st.text_input('Subject Folder Path', data_path)

subject_names = []
motion_names = []
motion_paths = []

motion_dict = {}

if os.path.exists(data_path_input):

    for dirpath, dirnames, filenames in os.walk(data_path_input):
        new_subject = os.path.split(dirpath)[-1]
        new_motions = {f for f in filenames if f.endswith('.pkl')}
        if new_motions:
            motion_dict[new_subject] = new_motions
            motion_names += new_motions
            motion_paths += {os.path.join(dirpath, m) for m in new_motions}
            subject_names += [new_subject]

    st.write(f'{len(subject_names)} subjects found.')
    st.write(f'{len(motion_names)} motions found.')

    motion_dict = dict(sorted(motion_dict.items(), key=lambda x: int(x[0])))

    motion_tree = [
        {
            'label': subject,
            'value': subject,
            'showCheckbox': False,
            'children': [
                {
                    'label': motion,
                    'value': motion,
                    'showCheckbox': False
                }
            for motion in motions]
        }
    for subject, motions in motion_dict.items()]

    tree_select(motion_tree)

    st.session_state['data_path'] = data_path_input
    st.session_state['motion_paths'] = dict(zip(motion_names, motion_paths))

    if 'feature_df' in st.session_state:
        del st.session_state['feature_df']

else:

    st.error('Subject Folder Path does not exist.')

    st.session_state['data_path'] = None
