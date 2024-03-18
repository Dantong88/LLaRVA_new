import json
import os
from tqdm import tqdm
import random

if __name__ == '__main__':
    root = '/home/niudt/LLaVA/process_dataset/large_scale_training/subset_ann/wotj'

    ann_file = os.listdir(root)
    train = 0
    val = 0
    info = {}
    info['val'] = {}

    train_ann = []
    val_ann = []

    for ann in tqdm(ann_file):
        s = 1
        subset = ann.split('-')[0]
        num = int(ann.split('-')[1])
        split = ann.split('-')[2].split('.')[0]
        if split == 'train':
            continue
        json_file = json.load(open(os.path.join(root, ann)))
        if split == 'train':
            train += num
            info['train'][subset] = num
            train_ann += json_file
        else:
            random.shuffle(json_file)
            val += num
            info['val'][subset] = num
            val_ann += json_file

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


    save_root = '/home/niudt/LLaVA/process_dataset/large_scale_training/exp_ann/exp4'
    os.makedirs(save_root, exist_ok=True)
    # train_save_path = os.path.join(save_root, 'train-{}.json'.format(len(train_ann)))
    # with open(train_save_path, 'w') as file:
    #     json.dump(train_ann, file)

    val_save_path = os.path.join(save_root, 'val-{}_whole.json'.format(len(val_ann)))
    with open(val_save_path, 'w') as file:
        json.dump(val_ann, file)

