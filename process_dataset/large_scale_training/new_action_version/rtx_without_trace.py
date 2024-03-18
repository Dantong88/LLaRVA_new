import os
import json
from tqdm import tqdm
import random
import argparse
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

def create_split(_root, subset, eps_list, start_idx):
    new_annotation = []
    annotation_image_root = os.path.join(_root, 'images')
    annotation_action_root = os.path.join(_root, 'new_actions')

    for eps in tqdm(eps_list):
        image_path = os.path.join(annotation_image_root, subset, eps)
        action_path = os.path.join(annotation_action_root, subset, eps)
        images = os.listdir(image_path)
        for image in images:
            _img_path = os.path.join(image_path, image)
            if subset == 'berkeley_rpt_converted_externally_to_rlds_new':
                _action_path = os.path.join(action_path, image.split('.')[0] + '.json')
            else:
                _action_path = os.path.join(action_path, eps + '_' + image.split('.')[0] + '.json')
            if (os.path.exists(_img_path) == False) or (os.path.exists(_action_path) == False):
                continue
            ann = json.load(open(_action_path))

            # interpret annotations
            robot_type = ann['robot_type']
            control_type = ann['control_type']
            prev_actions = ann['prev_actions']
            pred_actions = ann['pred_actions']
            instruction = ann['instruction']
            pred_num_step = ann['num_prediction_steps']

            s = 1
            # prev_action = []
            # for action in prev_actions[-5:]:
            #     prev_action += action

            prev_action = []
            for action in prev_actions[-5:]:
                prev_action.append(action)

            new_img_ann = {}
            # here process the action
            new_img_ann['id'] = str(start_idx)
            new_img_ann['image'] = _img_path[len(_root) + 1:]
            # create human conversation
            human = {}
            human[
                'value'] = '<image>\nYou are a {} robot using the {}. The task is \"{}\", and the previous five (including current) steps is {}, can you predict action of the next {} step?'.format(
                robot_type, control_type, instruction, prev_action, pred_num_step)
            human['from'] = 'human'
            gpt = {}
            gpt['from'] = 'gpt'
            gpt['value'] = 'The next action step: {}'.format(pred_actions)
            new_img_ann['conversations'] = [human, gpt]
            start_idx += 1
            new_annotation.append(new_img_ann)

        # if start_idx >= 1000:
        #     break

    return new_annotation, start_idx


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
    parser.add_argument('--idx', default=28, type=int, help='An input name')

    # Parse the arguments
    args = parser.parse_args()

    # root = json.load(open('/home/patrickwu/LLaVA/playground/data/llava_v1_5_mix665k.json'))
    _root = '/scratch/partial_datasets/llarva/rtx/v2'
    annotation_image_root = os.path.join(_root, 'images')
    annotation_action_root = os.path.join(_root, 'new_actions')
    subsets = os.listdir(annotation_action_root)
    ann_count = 1
    new_annotation = []
    Enable_Action = True
    train_start_idx = 1
    val_start_idx = 1

    for subset in subsets[args.idx: args.idx + 1]:
        if subset in DOWNSAMPLE:
            continue
        train_start_idx = 1
        val_start_idx = 1
        eps_list = os.listdir(os.path.join(annotation_image_root, subset))
        ratio = 0.8
        train_list = eps_list[: int(len(eps_list) * 0.8)]
        val_list = eps_list[int(len(eps_list) * 0.8):]

        train_split, train_start_idx = create_split(_root, subset, train_list, train_start_idx)
        val_split, val_start_idx = create_split(_root, subset, val_list, val_start_idx)
        save_root = './sub_anns'
        os.makedirs(save_root, exist_ok=True)
        train_save_path = os.path.join(save_root, '{}-{}-train.json'.format(subset, train_start_idx - 1))
        with open(train_save_path, 'w') as file:
            json.dump(train_split, file)

        val_save_path = os.path.join(save_root, '{}-{}-val.json'.format(subset, val_start_idx - 1))
        with open(val_save_path, 'w') as file:
            json.dump(val_split, file)








