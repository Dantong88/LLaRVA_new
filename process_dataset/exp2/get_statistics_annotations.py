import json
import os
from tqdm import tqdm
import random
import pandas as pd

if __name__ == '__main__':
    root = '/home/niudt/LLaVA/process_dataset/annotations'

    train_file = json.load(open(os.path.join(root, 'small_mixed_train_random_action_matrix_633136.json')))
    val_file = json.load(open(os.path.join(root, 'small_mixed_val_random_action_matrix_158284.json')))

    info = {}
    info_csv = []

    train_total = 0
    val_total = 0
    for ann in tqdm(train_file):
        s = 1
        subset = ann['image'].split('/')[1]
        if not subset in info:
            info[subset] = {}
            info[subset]['train'] = []
            info[subset]['val'] = []
        info[subset]['train'].append(ann)
        train_total += 1

    for ann in tqdm(val_file):
        s = 1
        subset = ann['image'].split('/')[1]
        if not subset in info:
            info[subset] = {}
        info[subset]['val'].append(ann)
        val_total += 1

    s = 1
    info_csv.append(['subset', 'train', 'val'])
    for subset in info:
        info_csv.append([subset, len(info[subset]['train']), len(info[subset]['val'])])
    info_csv.append(['all', train_total, val_total])

    headers = info_csv[0]
    df = pd.DataFrame(info_csv[1:], columns=headers)

    # Specify the filename
    filename = './exp2_info.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)



    #
    #     subset = ann.split('-')[0]
    #     num = int(ann.split('-')[1])
    #     split = ann.split('-')[2].split('.')[0]
    #     if not subset in info:
    #         info[subset] = {}
    #     info[subset][split] = num
    #     if split == 'train':
    #         train += num
    #     else:
    #         val += num
    #
    # s = 1
    # for subset in info:
    #     info_csv.append([subset, info[subset]['train'], info[subset]['val']])
    #
    # info_csv.append(['all', train, val])
    # s = 1
    #



