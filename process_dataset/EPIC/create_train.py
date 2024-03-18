import json
import os
from tqdm import tqdm
import random

if __name__ == '__main__':
    root = '/home/niudt/LLaVA/process_dataset/EPIC/annotations/dp'

    ann_file = os.listdir(root)
    train = 0
    val = 0
    info = {}
    info['train'] = {}
    info['val'] = {}

    train_ann = []
    val_ann = []

    for ann in tqdm(ann_file):
        s = 1
        subset = ann.split('_')[0]
        # num = int(ann.split('-')[1])
        split = ann.split('_')[2].split('_')[0]
        json_file = json.load(open(os.path.join(root, ann)))
        num = len(json_file)
        if split == 'train':
            train += num
            info['train'][subset] = num
            train_ann += json_file
        else:
            random.shuffle(json_file)
            val += num
            info['val'][subset] = num
            val_ann += json_file[:1000]

    s = 1
    # change the order
    train_count = 1
    val_count = 1
    for ann in tqdm(train_ann):
        ann['id'] = train_count
        train_count += 1
    for ann in tqdm(val_ann):
        ann['id'] = val_count
        val_count += 1

    assert (train_count - 1) == len(train_ann)
    assert (val_count - 1) == len(val_ann)


    save_root = './annotations_combine'
    os.makedirs(save_root, exist_ok=True)
    train_save_path = os.path.join(save_root, 'train-{}.json'.format(len(train_ann)))
    with open(train_save_path, 'w') as file:
        json.dump(train_ann, file)

    val_save_path = os.path.join(save_root, 'val-{}.json'.format(len(val_ann)))
    with open(val_save_path, 'w') as file:
        json.dump(val_ann, file)

    import pandas as pd

    info_stat = {}
    for subset in info['train']:
        if not subset in list(info_stat.keys()):
            info_stat[subset] = {}
        info_stat[subset]['train'] = info['train'][subset]

    for subset in info['val']:
        if not subset in list(info_stat.keys()):
            info_stat[subset] = {}
        info_stat[subset]['val'] = info['val'][subset]

    info = []
    for subset in info_stat:
        info.append([subset, info_stat[subset]['train'], info_stat[subset]['val']])
    # Convert the list into a DataFrame
    df = pd.DataFrame(info, columns=['subset', 'train', 'val'])

    # Save the DataFrame to a CSV file
    df.to_csv('./epic_stat.csv', index=False)
