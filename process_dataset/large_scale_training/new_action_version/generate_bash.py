import os


bash_script_content = ''

for i in range(37):
    if i == 14:
        continue
    current_command = 'source activate llava\n python /home/niudt/LLaVA/process_dataset/large_scale_training/new_action_version/rtx_without_trace.py --idx {}'.format(i)
    # Define the content to be written to the bash script
    bash_script_content = current_command

    # Define the filename of the bash script
    filename = './script/oxe-{}.sh'.format(i)

    # Open the file in write mode and write the content
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write('#!/bin/bash\n')  # Optional shebang to specify the bash interpreter
        file.write(bash_script_content + '\n')

    # Making the bash script executable (optional)
    import os
    os.chmod(filename, 0o755)
