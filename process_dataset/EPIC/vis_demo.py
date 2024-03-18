# this file is to visualize the demo

import os
import json
from tqdm import tqdm
import random
import argparse
import re
import matplotlib.pyplot as plt
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
        _p += s_a[0][3:]
        if withaction:
            _p += s_a[1]

    if withaction:
        _p += curr[0]
        _c = curr[1]
    else:
        _p += curr[0][3:]
        _c = [x + y for x, y in zip(curr[0][3:], curr[1][3:])]

    # round to 1e-4
    _p = [round(num, 4) for num in _p]
    _c = [round(num, 4) for num in _c]
    return _p, _c

def extract_number(s):
    return int(re.search(r'\d+', s).group())

def find_tj(tj_ann, idx, mode = 'right', dp_ratio=1):
    s = 1
    trajectory = []
    tjs = tj_ann[mode][idx:]
    s = 1
    for _p in tjs:
        if _p == [-1]:
            break

        trajectory.append([round(_p[0]), round(_p[1])])

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


def create_split(ek_instruction, _root, subset, eps_list, start_idx):
    new_annotation = []

    for eps in tqdm(eps_list):
        image_path = os.path.join(_root, subset, eps)
        images = [i for i in os.listdir(image_path) if i[-3:] == 'jpg']
        images = sorted(images, key=extract_number)
        tj_ann_path = os.path.join(image_path, 'trace_results.json')
        tj_ann = json.load(open(tj_ann_path))
        # get the instructions here

        for idx, image in enumerate(images):
            _img_path = os.path.join(image_path, image)

            # interpret annotations
            robot_type = 'ego person'
            control_type = 'hand control'
            instruction = ek_instruction[eps]

            # here get the trajectories
            right_hand_tj = find_tj(tj_ann, idx, mode='right_hands')
            left_hand_tj = find_tj(tj_ann, idx, mode='left_hands')



            new_img_ann = {}
            # here process the action
            new_img_ann['id'] = str(start_idx)
            new_img_ann['image'] = _img_path[len(_root):]
            # create human conversation
            human = {}
            human[
                'value'] = '<image>\nYou are a/an {} robot using the {}. The task is {}, can you predict the trajecorties of the hands?'.format(
                robot_type, control_type, instruction)
            human['from'] = 'human'
            gpt = {}
            gpt['from'] = 'gpt'
            gpt['value'] = 'Right hand trjectory: {}. Left hand trajectry: {}'.format(right_hand_tj, left_hand_tj)
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

            start_idx += 1
            new_annotation.append(new_img_ann)

        # if start_idx >= 1000:
        #     break

    return new_annotation, start_idx

def plot(points_r, points_l, image_path, title, save_path):

    # Assume image_path is the path to your image and points is your list of (x, y) tuples

    # Load and show the image
    img = plt.imread(image_path)
    plt.imshow(img)

    # Plot and connect the points
    # Unpack the list of points into x and y coordinates and plot them
    if not len(points_r) == 0:
        x_coords, y_coords = zip(*points_r)
        plt.plot(x_coords, y_coords, 'ro-')  # 'ro-' indicates red color, circle marker, and solid line

    if not len(points_l) == 0:
        x_coords, y_coords = zip(*points_l)
        plt.plot(x_coords, y_coords, 'bo-')

    # Set the title
    plt.title(title)

    # Save the plot as a JPEG file
    plt.savefig(save_path)

    # # Optionally display the plot
    # plt.show()

    plt.close()


def create_demo(ek_instruction, _root, subset, eps, dp_ratio):
    s = 1
    image_path = os.path.join(_root, subset, eps)
    images = [i for i in os.listdir(image_path) if i[-3:] == 'jpg']
    images = sorted(images, key=extract_number)
    tj_ann_path = os.path.join(image_path, 'trace_results.json')
    tj_ann = json.load(open(tj_ann_path))
    for idx, image in enumerate(tqdm(images)):
        s = 1
        _img_path = os.path.join(image_path, image)
        instruction = ek_instruction[eps]

        # here get the trajectories
        right_hand_tj = find_tj(tj_ann, idx, mode='right_hands', dp_ratio=dp_ratio)
        left_hand_tj = find_tj(tj_ann, idx, mode='left_hands',dp_ratio=dp_ratio)
        s = 1

        # create the images
        assert os.path.exists(_img_path)
        save_path = './demo_vis_dp/{}/{}/{}.jpg'.format(subset, eps, idx)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plot(points_r = right_hand_tj, points_l = left_hand_tj, image_path = _img_path, title = instruction, save_path = save_path)

DOWNSAMPLE = {
'add':	1/2,
'adjust': 1,
'apply': 1/5,
'attach': 1/2,
'bake':	1/2,
'break': 1/10,
'brush': 1/10,
'carry': 1/2,
'check': 1/10,
'choose': 1/10,
'close': 1/2,
'coat': 1/2,
'cook': 1/10,
'crush': 1/20,
'cut': 1/10,
'divide': 1/20,
'drink': 1/2,
'drop': 1/2,
'dry': 1/10,
'eat': 1/2,
'empty': 1/2,
'feel': 1/10,
'fill': 1/2,
'filter': 1/2,
'finish': 1/2,
'flatten': 1/10,
'flip': 1/10,
'fold': 1/2,
'form': 1/10,
'gather': 1/2,
'grate': 1/10,
'hang': 1/2,
'hold': 1/10,
'increase': 1/2,
'insert': 1/2,
'knead': 1/10,
'let-go': 1/2,
'lift':  1/2,
'lock': 1/2,
'look': 1/10,
'lower': 1/2,
'mark': 1/10,
'measure': 1/10,
'mix': 1/10,
'move': 1/2,
'open': 1/2,
'pat': 1/2,
'peel': 1/10,
'pour': 1/2,
'prepare': 1/10,
'press': 1/10,
'pull': 1/2,
'put': 1/2,
'remove': 1/2,
'rip': 1/2,
'roll': 1/10,
'rub': 1/10,
'scoop': 1/10,
'scrape': 1/10,
'screw': 1/10,
'scrub': 1/20,
'search': 1/10,
'season': 1/10,
'serve': 1/10,
'set': 1/10,
'shake': 1/10,
'sharpen': 1/10,
'slide': 1/10,
'smell': 1/2,
'soak': 1/2,
'sort': 1/10,
'spray': 1/2,
'sprinkle': 1/10,
'squeeze':  1/10,
'stab': 1/2,
'stretch': 1/10,
'switch':  1/10,
'take': 1/2,
'throw': 1/2,
'transition': 1/2,
'turn': 1/10,
'turn-down': 1/2,
'turn-off': 1/2,
'turn-on': 1/2,
'uncover': 1/2,
'unfreeze': 1/10,
'unlock': 1/2,
'unroll': 1/10,
'unscrew': 1/2,
'unwrap': 1/10,
'use': 1/2,
'wait': 1/20,
'wash':	1/10,
'water':	1/10,
'wear':	1/10,
'wrap':	1/10,
}


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')

    # Add arguments
    parser.add_argument('--idx', default=10, type=int, help='An input name')

    # Parse the arguments
    args = parser.parse_args()

    # root = json.load(open('/home/patrickwu/LLaVA/playground/data/llava_v1_5_mix665k.json'))
    _root = '/scratch/partial_datasets/llarva/epic_kitchen'
    subsets = os.listdir(_root)
    selected_demo_number = 2
    ek_instruction = json.load(open('./ek_eps.json'))

    ann_count = 1
    new_annotation = []
    Enable_Action = True
    train_start_idx = 1
    val_start_idx = 1
    for subset in subsets[args.idx:args.idx + 1]:
        dp_ratio = DOWNSAMPLE[subset]
        train_start_idx = 1
        val_start_idx = 1
        eps_list = os.listdir(os.path.join(_root, subset))
        random.shuffle(eps_list)


        for eps in eps_list[:selected_demo_number]:
            create_demo(ek_instruction, _root, subset, eps, dp_ratio)

        # train_split, train_start_idx = create_split(ek_instruction, _root, subset, train_list, train_start_idx)
        # val_split, val_start_idx = create_split(ek_instruction, _root, subset, val_list, val_start_idx)
        # save_root = './annotations'
        # os.makedirs(save_root, exist_ok=True)
        # train_save_path = os.path.join(save_root, '{}-{}-train.json'.format(subset, train_start_idx - 1))
        # with open(train_save_path, 'w') as file:
        #     json.dump(train_split, file)
        #
        # val_save_path = os.path.join(save_root, '{}-{}-val.json'.format(subset, val_start_idx - 1))
        # with open(val_save_path, 'w') as file:
        #     json.dump(val_split, file)











