# this file directly add trajectroy information to annotations



import json
import os
from tqdm import tqdm
import re
import math
def find_tj(tj_ann, idx, mode = 'ann'):
    trajectory = []
    tjs = tj_ann[mode][idx:]
    for _p in tjs:
        if _p == [-1]:
            continue

        try:
            trajectory.append([round(_p[0]), round(_p[1])])
        except:
            return []

    if len(trajectory) == 0:
        return []

    dp_ratio = 30/len(trajectory)
    stride = math.floor(1/dp_ratio)
    tj_new = []
    if stride <= 1:
        tj_new = trajectory
    else:
        start = 0
        while start < len(trajectory):
            tj_new.append(trajectory[start])
            start += stride
    trajectory = tj_new


    return trajectory


if __name__ == '__main__':
    ann_root = '/home/niudt/sub_anns'
    file_list = os.listdir(ann_root)
    for sub_file in file_list:
        # sub_file = 'val-36743.json'
        file_path = os.path.join(ann_root, sub_file)

        file = json.load(open(file_path))


        new_ann = []

        s = 1
        for ann in tqdm(file):
            s = 1
            idx = int(ann['image'].split('/')[3].split('.')[0])
            eps_id = 'episode_{}.json'.format(ann['image'].split('/')[2])
            subset_id = ann['image'].split('/')[1]
            tj_path = os.path.join('/home/niudt/dataset/trajectories', subset_id, eps_id)
            if not os.path.exists(tj_path):
                continue
            tj_file = json.load(open(tj_path))
            tj = find_tj(tj_file, idx, mode = 'ann')


            # change the prompt
            question = ann['conversations'][0]
            question['value'] = question['value'][: -1] + ' and the trajectories of the end effector?'

            answer = ann['conversations'][1]
            answer['value'] = answer['value'] + '. The trajectory: {}'.format(tj)
            new_ann.append(ann)

        s = 1
        # save
        save_root = '/home/niudt/LLaVA/process_dataset/large_scale_training/new_action_version/sub_anns_withtrace'
        os.makedirs(save_root, exist_ok=True)
        train_save_path = os.path.join(save_root, sub_file.split('.')[0] + '_{}.json'.format(len(new_ann)))
        with open(train_save_path, 'w') as file:
            json.dump(new_ann, file)


