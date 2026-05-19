# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import json
from argparse import ArgumentParser
import glob
import pandas as pd

parser = ArgumentParser(description="Training script parameters")
parser.add_argument('result_root')
args = parser.parse_args()

scenes = [
    'scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122'
]

cf = []
d2s = []
s2d = []
tr_time = []
fps = []
n_voxels = []

for scene in scenes:

    try:
        eval_path = glob.glob(f'{args.result_root}/*{scene}')[0] + '/test/mesh/results.json'
        with open(eval_path) as f:
            ret = json.load(f)
            cf.append(ret['overall'])
            d2s.append(ret['mean_d2s'])
            s2d.append(ret['mean_s2d'])
    except:
        cf.append(10)
        d2s.append(10)
        s2d.append(10)



def format_df_string(df):
    df = df.copy()
    df['scene'] = df['scene'].map(lambda s: s.rjust(15))
    df['d2s'] = df['d2s'].round(3)
    df['s2d'] = df['s2d'].round(3)
    df['cf-dist'] = df['cf-dist'].round(3)
    return df.to_string(index=False)

def add_avg_row(df):
    df_avg = df.mean(axis=0, numeric_only=True).to_frame().transpose()
    df_avg['scene'] = 'AVG'
    return pd.concat([df, df_avg], ignore_index=True)

df = pd.DataFrame({
    'scene': scenes,
    'cf-dist': cf,
    'd2s': d2s,
    's2d': s2d,
})
df = add_avg_row(df)

print(format_df_string(df))

