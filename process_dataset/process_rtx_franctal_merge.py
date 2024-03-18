import os
import json
from tqdm import tqdm
import random
def create_trace(img_idx, ann, word_coordinates):
    s = 1
    traces = []
    value = ''
    for idx, word in enumerate(ann['instruction_times']):
        if img_idx < int(word['end_frame'] + 1):
            act = 1
        else:
            act = 0
        coord = word_coordinates[word['word']]
        # value += word['word']
        # value += ','
        value += str([coord[0], coord[1], act])
        if not idx == len(ann['instruction_times']) - 1:
            value += ', '
        traces.append([coord[0], coord[1], act])
    value = value.strip()

    return value


def calculate_words_traj_mapping(ann):
    s = 1
    mapping = {}
    for word in ann['instruction_times']:
        s = 1
        tx = 0
        ty = 0
        for frame in range(int(word['start_frame']), int(word['end_frame'] + 1)):
            assert ann['trajectory'][frame][2] == frame
            tx += ann['trajectory'][frame][0]
            ty += ann['trajectory'][frame][1]
        tx = int(tx / (int(word['end_frame']) - int(word['start_frame']) + 1))
        ty = int(ty / (int(word['end_frame']) - int(word['start_frame']) + 1))
        s = 1
        mapping[word['word']] = (tx, ty)
    return mapping


def process_action(action_ann, withaction=False):
    s = 1
    prev = action_ann['prev_states_and_actions']
    curr = action_ann['current_state_and_action']
    _p = []
    _c = []
    for s_a in prev:
        _p.append(s_a[0][7:10] + s_a[0][4:7] + s_a[0][3:4])
        if withaction:
            _p += s_a[1]

    if withaction:
        _p += curr[0]
        _c = curr[1]
    else:
        _p.append(curr[0][7:10] + curr[0][4:7] + curr[0][3:4])
        _c = [x + y for x, y in zip(curr[0][7:10] + curr[0][4:7] + curr[0][3:4], curr[1][7:10] + curr[0][4:7] + curr[0][3:4])]

    # round to 1e-4
    new_p = []
    for sub_p in _p:
        new_p.append([round(num, 4) for num in sub_p])

    _p = new_p
    _c = [round(num, 4) for num in _c]
    return _p, _c

def create_split(tasks, save_path, mix = None):
    new_annotation = []
    ann_count = 1

    if mix != None:
        for item in tqdm(mix):
            s = 1
            try:
                item['image'] = os.path.join('/home/patrickwu/LLaVA/playground/data', item['image'])
            except:
                continue
            assert os.path.exists(item['image'])
            item['id'] = str(ann_count)

            ann_count += 1
            new_annotation.append(item)

    for episode in tqdm(tasks):
        ann = json.load(open(os.path.join(episode, 'rtx_ln_formatted.json')))
        # calcualte word coordinates mapping
        word_coordinates = calculate_words_traj_mapping(ann)
        episode_idx = episode.split('/')[-1].split('_')[1]
        image_root = os.path.join('/home/yuvan/rtx_trace_sample/rtx_images/fractal20220817_data', episode.split('/')[-2])
        episode_length = len(ann['trajectory'])
        s = 1
        for img_idx in range(episode_length):
            new_img_ann = {}
            img_path = os.path.join(image_root, '{}_{}.jpg'.format(episode_idx, img_idx))
            assert os.path.exists(img_path)
            # here process the action
            if Enable_Action:
                action_ann_path = os.path.join('/home/yuvan/rtx_trace_sample/fractal_action_json', '{}_{}.json'.format(episode_idx,img_idx))
                if not os.path.exists(action_ann_path):
                    continue
                action_ann = json.load(open(action_ann_path))
                previous, predict = process_action(action_ann)

            new_img_ann['id'] = str(ann_count)
            new_img_ann['image'] = '{}/{}_{}.jpg'.format(image_root.split('/')[-1],episode_idx, img_idx)
            # create human conversation
            human = {}
            human['value'] = '<image>\nYou are a franka robot using the end effector control. The task is {}, and the previous five (including current) steps is {}, can you predict the trajectory of the end effector and next step?'.format(task, previous)
            human['from'] = 'human'
            gpt = {}
            gpt['from'] = 'gpt'
            gpt['value'] = 'The trajectory: {}; The next step: {}'.format(create_trace(img_idx, ann, word_coordinates), predict)
            new_img_ann['conversations'] = [human, gpt]

            # if Enable_Action:
            #     human = {}
            #     human['value'] = 'Previous states: {}'.format(previous)
            #     human['from'] = 'human'
            #     gpt = {}
            #     gpt['from'] = 'gpt'
            #     gpt['value'] = 'Next states: {}'.format(predict)
            #     new_img_ann['conversations'].append(human)
            #     new_img_ann['conversations'].append(gpt)




            ann_count += 1
            new_annotation.append(new_img_ann)
    s = 1


    # save the file
    print(ann_count)


    save_root = os.path.dirname(save_path)
    os.makedirs(save_root, exist_ok=True)
    with open(save_path, 'w') as file:
        json.dump(new_annotation, file)



if __name__ == '__main__':
    template = json.load(open('/home/patrickwu/LLaVA/playground/data/llava_v1_5_mix665k.json'))
    annotation_folder = '/home/yuvan/rtx_trace_sample/rtx_trajectories_ln_format/fractal20220817_data/'
    tasks = os.listdir(annotation_folder)
    episode_list = []
    for task in tasks:
        episodes = os.listdir(os.path.join(annotation_folder, task))
        for episode in episodes:
            episode_list.append(os.path.join(annotation_folder, task, episode))


    Enable_Action = True
    # here need to shuffle the task and create episode level split
    random.shuffle(episode_list)

    train_split = int(len(episode_list) * 0.8)

    create_split(episode_list[:train_split], save_path = './annotations/feb29_train_{}_mixed.json'.format(train_split), mix = None)
    create_split(episode_list[train_split:],
                 save_path='./annotations/feb29_val_{}.json'.format(len(episode_list) - train_split))







