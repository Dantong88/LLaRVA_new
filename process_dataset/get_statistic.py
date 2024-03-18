import os
import json



if __name__ == '__main__':
    # path = '/scratch/partial_datasets/llarva/rtx/images'
    # subset = os.listdir(path)
    # info = []
    # for _s in subset:
    #     s = 1
    #     images_list = os.listdir(os.path.join(path, _s))
    #     info.append([_s, len(images_list)])
    # s = 1
    # import pandas as pd
    #
    # # Convert the list into a DataFrame
    # df = pd.DataFrame(info, columns=['subset', 'number'])
    #
    # # Save the DataFrame to a CSV file
    # df.to_csv('./rtx_small_mixed.csv', index=False)

    train_set = json.load(open('/home/niudt/LLaVA/process_dataset/annotations/small_mixed_val.json'))

    s = 1

