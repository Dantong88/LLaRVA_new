import os
import json
import random
from tqdm import tqdm



if __name__ == '__main__':
    val_file = json.load(open('/home/niudt/LLaVA/process_dataset/annotations/exp1/feb29_val_5301.json'))


    our_questionset = []
    for i, sample in enumerate(tqdm(val_file)):
        current_question = {}
        current_question['image'] = sample['image']
        # # filter the episode
        # current_episode = sample['image'].split('/')[0]
        # current_episode_index = sample['image'].split('/')[1].split('_')[0]
        # if current_episode != action:
        #     continue
        # if current_episode_index != str(index):
        #     continue
        # s = 1
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
        our_questionset.append(current_question)

    s = 1

    # Path to your JSONL file
    file_path = './demo_{}_questions.jsonl'.format(len(our_questionset))

    # Write the dictionaries to a JSONL file
    with open(file_path, 'w') as file:
        for d in our_questionset:
            json_line = json.dumps(d) + '\n'  # Convert dict to JSON string and add newline
            file.write(json_line)

