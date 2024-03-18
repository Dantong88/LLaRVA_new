# this file directly add trajectroy information to annotations



import json
import os
from tqdm import tqdm
import re
def find_tj(tj_ann, idx, mode = 'ann', dp_ratio = 1):
    trajectory = []
    tjs = tj_ann[mode][idx:]
    for _p in tjs:
        if _p == [-1]:
            break

        try:
            trajectory.append([round(_p[0]), round(_p[1])])
        except:
            return []

    if len(trajectory) == 0:
        return []

    stride = int(1/dp_ratio)
    tj_new = []
    if len(trajectory) <= stride:
        tj_new.append(trajectory[0])
    else:
        start = 0
        while start < len(trajectory):
            tj_new.append(trajectory[start])
            start += stride
    trajectory = tj_new


    return trajectory


if __name__ == '__main__':
    ann_root = '/home/niudt/LLaVA/process_dataset/large_scale_training/exp_ann/exp4'
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
            tj = find_tj(tj_file, idx, mode = 'ann', dp_ratio = 1)


            # change the prompt
            question = ann['conversations'][0]
            question['value'] = question['value'][: -1] + ' and the trajectories of the end effector?'

            # add the quaotes
            pattern = re.compile(r"The task is (.*?), and the previous")
            question['value'] = pattern.sub(lambda m: f'the task is "{m.group(1)}", and the previous',
                                            question['value'])

            answer = ann['conversations'][1]
            answer['value'] = answer['value'].replace("The next step", "The next action step")

            answer = ann['conversations'][1]
            answer['value'] = answer['value'] + '. The trajectory: {}'.format(tj)


            s = 1


            new_ann.append(ann)

        s = 1
        # save
        save_root = '/home/niudt/LLaVA/process_dataset/large_scale_training/exp_ann/exp5'
        os.makedirs(save_root, exist_ok=True)
        train_save_path = os.path.join(save_root, sub_file.split('.')[0] + '_{}.json'.format(len(new_ann)))
        with open(train_save_path, 'w') as file:
            json.dump(new_ann, file)


