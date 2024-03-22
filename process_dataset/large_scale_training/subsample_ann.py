import json
import random
import os
if __name__ == '__main__':
    file = json.load(open('/home/niudt/LLaVA/process_dataset/large_scale_training/new_action_version/exp4/train-34053947.json'))
    s = 1
    random.shuffle(file)

    file = file[:int(0.25 * len(file))]
    save_root = '/home/niudt/LLaVA/process_dataset/large_scale_training/new_action_version/exp4'
    os.makedirs(save_root, exist_ok=True)
    train_save_path = os.path.join(save_root, 'quarter-train-{}.json'.format(len(file)))
    with open(train_save_path, 'w') as file_:
        json.dump(file, file_)

