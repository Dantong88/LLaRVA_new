import json
import random
from tqdm import tqdm
# Replace 'your_file.jsonl' with the path to your JSONL file
answer_file = '/home/niudt/LLaVA/process_dataset/exp4/demos_resuls/14441_answers.jsonl'

# Open the JSONL file
ans = []
with open(answer_file, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Parse the JSON object
        template = json.loads(line)
        # Now you can work with the json_object as a dictionary
        ans.append(template)

s = 1

# questions_file = '/home/niudt/LLaVA/process_dataset/eval/50rtx_questions.jsonl'
#
# # Open the JSONL file
# ques = []
# with open(answer_file, 'r') as file:
#     # Iterate over each line in the file
#     for line in file:
#         # Parse the JSON object
#         template = json.loads(line)
#         # Now you can work with the json_object as a dictionary
#         ques.append(template)
#
# s = 1
