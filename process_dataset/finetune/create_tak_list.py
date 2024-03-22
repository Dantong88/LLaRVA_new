import os
import json
import glob
from tqdm import tqdm
if __name__ == '__main__':
    info = {}
    subset = 'berkeley_rpt_converted_externally_to_rlds_new'
    root_path = os.path.join('/scratch/partial_datasets/llarva/rtx/v2/new_actions', subset)

    episode_list = os.listdir(root_path)

    for episode in tqdm(episode_list):
        random_image = glob.glob(os.path.join(root_path, episode, '*.json'))[0]
        file = json.load(open(random_image))

        task = file['instruction']

        if not task in info:
            info[task] = []

        info[task].append(episode)

    s = 1
    save_path = './rpt_task_list.json'
    with open(save_path, 'w') as file:
        json.dump(info, file)
