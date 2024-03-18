import json
import os
from tqdm import tqdm
import shutil
if __name__ == '__main__':
    # file = json.load(open('/home/niudt/LLaVA/process_dataset/annotations/franctal_train.json'))
    file = json.load(open('/home/niudt/LLaVA/process_dataset/annotations/feb29_val_5301.json'))
    s = 1
    save_root = '/home/niudt/dataset/franctal_llarva'
    for ann in tqdm(file):
        s = 1
        print(ann['image'])
        print(ann['conversations'])
        print('\n')
        # img_path = ann['image']
        # t_path = os.path.join(save_root, img_path.split('/')[-2], img_path.split('/')[-1])
        # s = 1
        # # create the dir
        # os.makedirs(os.path.dirname(t_path), exist_ok=True)
        # shutil.copy(img_path, t_path)

