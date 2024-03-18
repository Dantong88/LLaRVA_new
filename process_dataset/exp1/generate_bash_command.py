import os


if __name__ == '__main__':
    root = '/home/niudt/LLaVA/process_dataset/exp1/demos_val'

    files = os.listdir(root)

    for f in files:
        name = f.split('_')[0]
        question_file = os.path.join(root, f)
        save_path = '/home/niudt/LLaVA/process_dataset/exp1/demos_resuls/{}_answers.jsonl'.format(name)

        print('python /home/niudt/LLaVA/llava/eval/model_vqa.py --model-path /home/niudt/LLaVA/checkpoints/merge/llava-v1.5-7b-exp1 --question-file \'{}\' --answers-file \'{}\''.format(question_file, save_path))
        print('\n')
