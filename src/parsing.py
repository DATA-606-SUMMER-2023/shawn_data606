import os
import regex
import pickle

from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

def parse_asf(asf_file):
    """
    Parses asf files.
    asf files represent bone data at rest. This includes:
        order - The order of the x, y and z axes. Relevant for rotations and know which one points up.
        axis - The axis of rotation for a bone in euler angles.
        position - the xyz coordinates of a joint, applies only to root
        orientation - the euler angles for the rotation of a joint
        direction - the direction the bone extends outward
        length - the length of the bone, often indicated where the child joint begins
        dof - the degrees of freedom, their limits and their order
    """

    with open(asf_file, 'r') as asf:

        content = asf.read()

        # Remove comments
        content = regex.sub('(#.*\n|\n#.*$)', '', content)

        pattern = '(?::(?P<key>[\S]+)\s+(?P<value>[^:]+))*'
        capture = regex.match(pattern, content).capturesdict()
        parse_dict = dict(zip(capture['key'], capture['value']))

        root_pattern = (
            'order(?: (?P<order>\w*))*\s+'
            'axis (?P<axis>\w*)\s+'
            'position(?: (?P<position>[\d.e-]+))*\s+'
            'orientation(?: (?P<orientation>[\d.e-]+))*\s+'
        )
        parse_dict['root'] = regex.match(root_pattern, parse_dict['root']).capturesdict()

        root_dict = parse_dict['root']
        root_dict = {
            'order': [i.lower() for i in root_dict['order']],
            'axis': root_dict['axis'][0],
            'position': np.array([float(i) for i in root_dict['position']]),
            'orientation': np.array([float(i) for i in root_dict['orientation']])
        }

        bonedata_pattern = '(?:begin(?P<bone>[\w\W]*?)\n\s+end\s+)*'
        parse_dict['bonedata'] = regex.match(bonedata_pattern, parse_dict['bonedata']).capturesdict()['bone']

        bone_pattern = (
            '\s+'
            'id (?P<id>\d+)\s+'
            'name (?P<name>\w*)\s+'
            'direction (?:(?P<direction>[\d.e-]+) +)*\s+'
            'length (?P<length>[\d.e-]+)\s+'
            'axis (?:(?P<axis>[\d.e-]+) +)*(?P<axis_order>\w+)\s*'
            '(?:dof (?:(?P<dof>\w+) *)*)?\s*'
            '(?:limits (?:\((?P<limits>[\w\W]*?)\)\s*)*)?'
        )
        parse_dict['bonedata'] = [regex.match(bone_pattern, i).capturesdict() for i in parse_dict['bonedata']]

        hierarchy_pattern = 'begin\s+(?:(?P<hierarchy>.*)\s+)*end'
        parse_dict['hierarchy'] = [
            {
                'parent': i.split(' ')[0],
                'children': i.split(' ')[1:]
            }
            for i in regex.match(hierarchy_pattern, parse_dict['hierarchy']).capturesdict()['hierarchy']
        ]

        hierarchy_map = {i['parent']: i['children'] for i in parse_dict['hierarchy']}

        for bone in parse_dict['bonedata']:

            bone['id'] = int(bone['id'][0])
            bone['name'] = bone['name'][0]
            bone['direction'] = np.array([float(i) for i in bone['direction']])
            bone['length'] = float(bone['length'][0])
            bone['axis'] = np.array([float(i) for i in bone['axis']])
            bone['axis_order'] = bone['axis_order'][0]

            if 'dof' not in bone:
                bone['dof'] = []

            if 'limits' in bone:
                bone['limits'] = [tuple(float(j) for j in i.split(' ')) for i in bone['limits']]
            else:
                bone['limits'] = []

            bone['children'] = hierarchy_map.get(bone['name'])
            if bone['children'] is None:
                bone['children'] = []

    return parse_dict

def parse_amc(amc_file):
    """
    Parses amc files.
    amc files each represent one trial, and the motions applied to each bone for each frame in that trial.
    The numbers of a row correspond the the dof specified in the asf file.
    """

    with open(amc_file, 'r') as amc:

        content = amc.read()

        # Remove comments
        content = regex.sub('(#.*\n|\n#.*$)', '', content)
        content = regex.sub('\n$', '', content)
        split = regex.split('\n\d+\n', content)
        info, motion_text_list = split[0], split[1:]
        
        pattern = '(?P<key>[^ ]+)(?: (?P<values>[^ ]+))*'
        motions = []

        for m in motion_text_list:
            motion_dict = {}
            for line in m.split('\n'):
                capture = regex.match(pattern, line).capturesdict()
                motion_dict |= {capture['key'][0]: capture['values']}
            motions.append(motion_dict)

        return {'info': info, 'poses': motions}

def get_rest_df(asf_dict):
    """
    Forms parsed asf data into a pandas DataFrame representing the rest pose.

    Arguments:
    asf_dict -- output from parse_asf()
    """

    hierarchy_dict = {e['parent']: e['children'] for e in asf_dict['hierarchy']}
    
    root_dict = asf_dict['root']
    root_dict = {
        'root': {
            'position': np.array([0., 0., 0.]),
            'rotation': Rotation.from_euler('xyz', root_dict['orientation'], degrees=True).as_matrix(),
            'direction': np.array([0., 0., 0.,]),
            'axis': np.array([0., 0., 0.]),
            'length': 0,
            'dof': [i.lower() for i in root_dict['order']],
            'limits': [],
            'children': hierarchy_dict['root']
        }
    }

    bone_dict = root_dict | {
        b['name']: {
            'position': None,
            'rotation': None,
            'direction': b['direction'],
            'axis': b['axis'],
            'length': b['length'],
            'dof': b['dof'],
            'limits': b['limits'],
            'children': b['children']
        } for b in asf_dict['bonedata']
    }

    def define_bones(bone, offset):
        bone['position'] = offset
        bone['rotation'] = np.array([[1., 0., 0.], [0., 1., 0.,], [0., 0., 1.]])
        for c in bone['children']:
            child_bone = bone_dict[c]
            new_offset = offset + (bone['direction'] * bone['length'])
            define_bones(child_bone, offset=new_offset)

    define_bones(bone_dict['root'], root_dict['root']['position'])

    return pd.DataFrame(root_dict | bone_dict).T

def get_motion_df(amc_dict):
    """
    Forms parsed asf data into a pandas DataFrame representing keyframes of motion for one trial.

    Arguments:
    amc_dict -- output from parse_amc()
    """

    pose_list = amc_dict['poses']
    pose_list = [
        {k: np.array([float(i) for i in v]) for k, v in pose.items()}
        for pose in pose_list
    ]

    return pd.DataFrame(pose_list).T

def subject_to_dfs(subject_folder_path, amc_files='all', amc_limit=None):
    """
    Takes a folder containing an asf file and a bunch of amc files and compiles a rest dataframes and set of motion dataframes.
    The folder represents a subject, and the motion dataframes each represent a trial for that subject.
    
    Arguments:
    subject_folder_path -- the folder containing asf and amc files
    
    Keyword arguments:
    amc_files -- list of amc file names withing the folder to parse. 'all' means all amc files in the folder will be parsed.
    """

    subject_files = os.listdir(subject_folder_path)
    asf_path = subject_folder_path + '/' + next(f for f in subject_files if f.endswith('.asf'))

    if amc_files == 'all':
        amc_paths = [os.path.join(subject_folder_path, f) for f in subject_files if f.endswith('.amc')]
    else:
        amc_paths = [os.path.join(subject_folder_path, f) + '/' + f for f in amc_files]
    
    if amc_limit is not None:
        amc_paths = amc_paths[:amc_limit]

    rest_df = get_rest_df(parse_asf(asf_path))
    motion_dfs = [get_motion_df(parse_amc(amc_path)) for amc_path in amc_paths]

    return rest_df, motion_dfs
