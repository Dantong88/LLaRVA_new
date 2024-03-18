import os
import json
from tqdm import tqdm
import random
def create_trace(img_idx, ann, word_coordinates):
    s = 1
    traces = []
    value = ''
    for word in ann['instruction_times']:
        if img_idx < int(word['end_frame'] + 1):
            act = 1
        else:
            act = 0
        coord = word_coordinates[word['word']]
        value += word['word']
        value += ' '
        value += str([coord[0], coord[1], act])
        value += ' '
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
    new_annotation = json.load(open('../annotations/franctal_val.json'))

    # Splitting the list into 70% and 30% parts
    split_index = int(len(new_annotation) * 0.999)
    trainset = new_annotation[:split_index]
    valset = new_annotation[split_index:]

    save_root = '../annotations'
    os.makedirs(save_root, exist_ok=True)
    # train_save_path = os.path.join(save_root, 'franctal_train.json',)
    # with open(train_save_path, 'w') as file:
    #     json.dump(trainset, file)

    val_save_path = os.path.join(save_root, 'franctal_val_dummy.json', )
    with open(val_save_path, 'w') as file:
        json.dump(valset, file)






