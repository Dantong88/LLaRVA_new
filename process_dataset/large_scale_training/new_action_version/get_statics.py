import json
import os
from tqdm import tqdm
import random
import pandas as pd

if __name__ == '__main__':
    root = '/home/niudt/LLaVA/process_dataset/large_scale_training/new_action_version/sub_anns'

    ann_file = os.listdir(root)
    train = 0
    val = 0
    info = {}

    train_ann = []
    val_ann = []


    info_csv = []
    info_csv.append(['idx', 'subset', 'train', 'val'])

    for ann in tqdm(ann_file):
        s = 1
        subset = ann.split('-')[0]
        num = int(ann.split('-')[1])
        split = ann.split('-')[2].split('.')[0]
        if not subset in info:
            info[subset] = {}
        info[subset][split] = num
        if split == 'train':
            train += num
        else:
            val += num

    s = 1
    for idx, subset in enumerate(info):
        info_csv.append([idx + 1, subset, info[subset]['train'], info[subset]['val']])

    info_csv.append([idx + 2, 'all', train, val])
    s = 1

    headers = info_csv[0]
    df = pd.DataFrame(info_csv[1:], columns=headers)

    # Specify the filename
    filename = './rtx_info_withouttj.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


