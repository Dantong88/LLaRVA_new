import json
import random
from tqdm import tqdm
# Replace 'your_file.jsonl' with the path to your JSONL file
file_path = '/home/patrickwu/LLaVA/playground/data/coco2014_val_qa_eval/qa90_questions.jsonl'

# Open the JSONL file
with open(file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Parse the JSON object
        template = json.loads(line)
        # Now you can work with the json_object as a dictionary
        print(template)
        break

training_file = json.load(open('../franctal_train_add_action_v2.json'))

s = 1

sample_num = 50

sample_subset = sampled_elements = random.sample(training_file, sample_num)

our_questionset = []
for i, sample in enumerate(tqdm(sample_subset)):
    current_question = {}
    current_question['image'] = sample['image']
    current_question['question_id'] = i
    current_question['category'] = 'conv'

    # for question
    for i, conv in enumerate(sample['conversations']):
        if i<2:
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
file_path = './50rtx_questions_add_action.jsonl'

# Write the dictionaries to a JSONL file
with open(file_path, 'w') as file:
    for d in our_questionset:
        json_line = json.dumps(d) + '\n'  # Convert dict to JSON string and add newline
        file.write(json_line)

