import os
import json

if __name__ == '__main__':
    org_file = json.load(open('/home/niudt/LLaVA/process_dataset/annotations/small_mixed_all_random_action_matrix_3058195.json'))

    s = 1
    for ann in org_file:
        image_path = ann['']

