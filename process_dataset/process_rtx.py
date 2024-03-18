import os
import json
from tqdm import tqdm

def create_trace(img_idx, ann, word_coordinates):
    s = 1
    traces = []
    for word in ann['instruction_times']:
        if img_idx < int(word['end_frame'] + 1):
            act = 1
        else:
            act = 0
        coord = word_coordinates[word['word']]
        traces.append([coord[0], coord[1], act])

    return traces


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



if __name__ == '__main__':
    template = json.load(open('/home/patrickwu/LLaVA/playground/data/llava_v1_5_mix665k.json'))
    annotation_folder = '/home/yuvan/rtx_trace_sample/rtx_trajectories_ln_format/fractal20220817_data/'
    tasks = os.listdir(annotation_folder)
    ann_count = 1
    new_annotation = []
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
                s = 1
                new_img_ann['id'] = str(ann_count)
                new_img_ann['image'] = img_path
                # create human conversation
                human = {}
                human['from'] = 'human'
                human['value'] = '<image>\n{}'.format(task)
                s = 1
                gpt = {}
                gpt['from'] = 'gpt'
                gpt['value'] = task
                new_img_ann['conversations'] = [human, gpt]
                new_img_ann['traces'] = create_trace(img_idx, ann, word_coordinates)
                s = 1
                ann_count += 1
                new_annotation.append(new_img_ann)
    s = 1
    # save the file
    print(ann_count)
    with open('./franctal_train.json', 'w') as file:
        json.dump(new_annotation, file)





