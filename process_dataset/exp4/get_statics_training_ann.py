import json
import os
from tqdm import tqdm

if __name__ == '__main__':
    file = json.load(open('/home/niudt/LLaVA/process_dataset/large_scale_training/exp_ann/exp4/quater-train-7087524.json'))


    info = {}

    for ann in file:
        image = ann['image']
        split = image.split('/')[1]
        if not split in info:
            info[split] = 0
        info[split] += 1

    s = 1
