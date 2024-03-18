import json
import pickle
import pickle
from tqdm import tqdm
import shutil
import os
import pandas as pd
import numpy as np


if __name__ == '__main__':
    train_annotation_path = '/home/niudt/mae-cross-dev-r2d2/prepare_the_dataset/sample_videos/EPIC_100_train.csv'
    val_annotation_path = '/home/niudt/mae-cross-dev-r2d2/prepare_the_dataset/sample_videos/EPIC_100_validation.csv'
    stat = {}
    train_annotations = pd.read_csv(train_annotation_path).values
    val_annotations = pd.read_csv(val_annotation_path).values
    annotations = np.concatenate((train_annotations, val_annotations), axis=0)

    s = 1
    for ann in tqdm(annotations):
        idx = ann[0]
        stat[idx] = ann[8]

    with open('./ek_eps.json', 'w') as f:
        json.dump(stat, f)
