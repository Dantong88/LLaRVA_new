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


if __name__ == '__main__':
    root = json.load(open('/home/patrickwu/LLaVA/playground/data/llava_v1_5_mix665k.json'))
    _root = '/scratch/partial_datasets/llarva/rtx/v1'
    annotation_image_root = os.path.join('/scratch/partial_datasets/llarva/rtx/v1', 'images')
    annotation_action_root = os.path.join('/scratch/partial_datasets/llarva/rtx/v1', 'actions')
    subsets = os.listdir(annotation_image_root)
    ann_count = 1
    new_annotation = []
    Enable_Action = True
    for subset in subsets:
        subset_count = 1
        image_path = os.path.join(annotation_image_root, subset)
        action_path = os.path.join(annotation_action_root, subset)
        images = os.listdir(image_path)
        for image in tqdm(images):
            _img_path = os.path.join(image_path, image)
            _action_path = os.path.join(action_path, image.split('.')[0]+'.json')
            if (os.path.exists(_img_path) == False) and (os.path.exists(_action_path) == False):
                continue
            ann = json.load(open(_action_path))

            # interpret annotations
            robot_type = ann['robot_type']
            control_type = ann['control_type']
            prev_actions = ann['prev_actions']
            pred_actions = ann['pred_actions']
            instruction = ann['instruction']

            s = 1
            # prev_action = []
            # for action in prev_actions[-5:]:
            #     prev_action += action

            prev_action = []
            for action in prev_actions[-5:]:
                prev_action.append(action)


            new_img_ann = {}
            # here process the action
            new_img_ann['id'] = str(ann_count)
            new_img_ann['image'] = _img_path[len(_root) + 1:]
            # create human conversation
            human = {}
            human['value'] = '<image>\nYou are a {} robot using the {}. The task is {}, and the previous five (including current) steps is {}, can you predict action of the next step?'.format(robot_type, control_type, instruction, prev_action)
            human['from'] = 'human'
            gpt = {}
            gpt['from'] = 'gpt'
            gpt['value'] = 'The next step: {}'.format(pred_actions)
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
            subset_count += 1
            new_annotation.append(new_img_ann)

            if subset_count >= 60000:
                break
    s = 1
    # save the file
    print(ann_count)

    # Shuffling the list randomly
    random.shuffle(new_annotation)

    # Splitting the list into 70% and 30% parts
    split_index = int(len(new_annotation) * 0.8)
    trainset = new_annotation[:split_index]
    valset = new_annotation[split_index:]

    save_root = './annotations'
    os.makedirs(save_root, exist_ok=True)
    train_save_path = os.path.join(save_root, 'small_mixed_train_random_action_matrix_{}.json'.format(len(trainset)),)
    with open(train_save_path, 'w') as file:
        json.dump(trainset, file)

    val_save_path = os.path.join(save_root, 'small_mixed_val_random_action_matrix_{}.json'.format(len(valset)), )
    with open(val_save_path, 'w') as file:
        json.dump(valset, file)






