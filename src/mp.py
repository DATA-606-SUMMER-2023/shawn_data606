import multiprocessing
from multiprocessing import Pool

import os

def animation_pickle(asf_file, amc_file):

    import os
    import re

    import pandas as pd
    import numpy as np

    import plotly.graph_objects as go
    import plotly.express as px
    # from tqdm.auto import tqdm

    # My modules
    import parsing
    import posing
    import plotting

    asf_dict = parsing.parse_asf(asf_file)
    amc_dict = parsing.parse_amc(amc_file)

    rest_df = parsing.get_rest_df(asf_dict)
    motion_df = parsing.get_motion_df(amc_dict)

    animation_df = pd.concat([posing.apply_pose(rest_df, motion_df[i]) for i in motion_df.columns], keys=motion_df.columns)
    animation_df.index.names = ['frame', 'joint']

    filename = amc_file.replace('.amc', '.pkl')
    
    animation_df.to_pickle(filename)
    print(f'Saved {filename}')

if __name__ == '__main__':

    subject_path = 'data/subjects'

    args = []

    for path, dirs, files in os.walk(subject_path):
        amcs = [f for f in files if f.endswith('.amc')]
        args += [
            (
                os.path.join(path, amc.split('_')[0] + '.asf'), 
                os.path.join(path, amc)
            ) for amc in amcs
        ]

    print(f'Processing {len(args)} files...')

    with Pool(multiprocessing.cpu_count() - 3) as p:
        p.starmap(animation_pickle, args)

    print('Done!')