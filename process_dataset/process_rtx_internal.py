import os
import json
from tqdm import tqdm

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
        _p += s_a[0]
        if withaction:
            _p += s_a[1]

    if withaction:
        _p += curr[0]
        _c = curr[1]
    else:
        _p += curr[0]
        _c = [x + y for x, y in zip(curr[0], curr[1])]
    return _p, _c


if __name__ == '__main__':
    template = json.load(open('/home/patrickwu/LLaVA/playground/data/llava_v1_5_mix665k.json'))
    annotation_folder = '/home/yuvan/rtx_trace_sample/rtx_trajectories_ln_format/fractal20220817_data/'
    tasks = os.listdir(annotation_folder)
    ann_count = 1
    new_annotation = []
    Enable_Action = True
    for task in tqdm(tasks):
        path = os.path.join(annotation_folder, task)
        s = 1
        episodes = os.listdir(path)
        s = 1
        for episode in episodes:
            ann = json.load(open(os.path.join(path, episode, 'rtx_ln_formatted.json')))
            # calcualte word coordinates mapping
            word_coordinates = calculate_words_traj_mapping(ann)



            episode_idx = episode.split('_')[1]
            image_root = os.path.join('/home/yuvan/rtx_trace_sample/rtx_images/fractal20220817_data', task)
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
                human['value'] = '<image>\n{}'.format(task)
                human['from'] = 'human'
                gpt = {}
                gpt['from'] = 'gpt'
                gpt['value'] = create_trace(img_idx, ann, word_coordinates)
                new_img_ann['conversations'] = [human, gpt]

                if Enable_Action:
                    human = {}
                    human['value'] = 'Previous states: {}'.format(previous)
                    human['from'] = 'human'
                    gpt = {}
                    gpt['from'] = 'gpt'
                    gpt['value'] = 'Next states: {}'.format(predict)
                    new_img_ann['conversations'].append(human)
                    new_img_ann['conversations'].append(gpt)




                ann_count += 1
                new_annotation.append(new_img_ann)
    s = 1
    # save the file
    print(ann_count)
    with open('./franctal_train_add_action_merge.json', 'w') as file:
        json.dump(new_annotation, file)





