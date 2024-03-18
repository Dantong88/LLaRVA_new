import os
import json
import random
from tqdm import tqdm



if __name__ == '__main__':
    val_file = json.load(open('/home/niudt/LLaVA/process_dataset/large_scale_training/exp_ann/exp4/val-7104523_whole.json'))

    episode = {}


    our_questionset = []
    for i, sample in enumerate(tqdm(val_file)):
        current_question = {}
        current_question['image'] = sample['image']
        # filter the episode
        current_episode = sample['image'].split('/')[-2]
        current_episode_index = sample['image'].split('/')[-1].split('_')[0]

        episode_name = current_episode
        if not episode_name in episode:
            episode[episode_name] = []
        current_question['question_id'] = i
        current_question['category'] = 'conv'

        # for question
        for i, conv in enumerate(sample['conversations']):
            if conv['from'] == 'human':
                text = conv['value'].split('<image>\n')[1]
                current_question['text'] = text

            else:
                text = conv['value']
                current_question['gt'] = text
        episode[episode_name].append(current_question)

    s = 1

    selected_num = 30
    selected_index = [random.randint(0, len(episode)) for _ in range(selected_num)]
    for idx, eps in enumerate(episode):
        if idx in selected_index:
            s = 1
            # save
            # Path to your JSONL file
            file_path = './demos_val/{}_questions.jsonl'.format(eps)

            # Write the dictionaries to a JSONL file
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                for d in episode[eps]:
                    json_line = json.dumps(d) + '\n'  # Convert dict to JSON string and add newline
                    file.write(json_line)

