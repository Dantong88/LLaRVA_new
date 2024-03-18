import os
import json
import random
from tqdm import tqdm



if __name__ == '__main__':
    # action_exmaple = {}
    # action_exmaple['robot_type'] = '' # see google sheet
    # action_exmaple['control_type'] = '' # end effector control / joint control
    # action_exmaple['interpretation'] = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'gripper closeness'] # this should be same dimension size with action
    # action_exmaple['prev_actions'] = [[],[],[],[]] # number round to 1e-4
    # action_exmaple['pred_actions'] = [] # number round to 1e-4


    action = 'pick green jalapeno chip bag from bottom drawer and place on counter'
    index = 9677
    episode_path = os.path.join('/home/niudt/dataset/franctal_llarva', action)
    mode = 'pixel'



    # Replace 'your_file.jsonl' with the path to your JSONL file
    file_path = '/home/patrickwu/LLaVA/playground/data/coco2014_val_qa_eval/qa90_questions.jsonl'


    training_file = json.load(open('../franctal_train_add_action_v2.json'))


    our_questionset = []
    for i, sample in enumerate(tqdm(training_file)):
        current_question = {}
        current_question['image'] = sample['image']
        # filter the episode
        current_episode = sample['image'].split('/')[0]
        current_episode_index = sample['image'].split('/')[1].split('_')[0]
        if current_episode != action:
            continue
        if current_episode_index != str(index):
            continue
        s = 1
        current_question['question_id'] = i
        current_question['category'] = 'conv'

        # for question
        for i, conv in enumerate(sample['conversations']):
            if mode == 'pixel':
                if i >= 2:
                    continue
                if conv['from'] == 'human':
                    text = conv['value'].split('<image>\n')[1]
                    current_question['text'] = text

                else:
                    text = conv['value']
                    current_question['gt'] = text
            else:
                if i < 2:
                    continue
                if conv['from'] == 'human':
                    text = conv['value']
                    current_question['text'] = text

                else:
                    text = conv['value']
                    current_question['gt'] = text
        our_questionset.append(current_question)

    s = 1

    # Path to your JSONL file
    file_path = './demo_{}.jsonl'.format(mode)

    # Write the dictionaries to a JSONL file
    with open(file_path, 'w') as file:
        for d in our_questionset:
            json_line = json.dumps(d) + '\n'  # Convert dict to JSON string and add newline
            file.write(json_line)

