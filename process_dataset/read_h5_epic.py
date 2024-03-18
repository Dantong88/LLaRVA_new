import h5py
import json
import pickle
import pickle
from tqdm import tqdm
import shutil
import os
import pandas as pd
import numpy as np

# Now you can use 'your_data' as a normal Python object.

def save_episode(ann, save_id, save_root):
    index = 1
    start_frame = ann[6]
    stop_frame = ann[7]
    stride = 1
    if stride >= 1:
        for _frame in range(start_frame, stop_frame + 1, stride):
            path = os.path.join('/datasets/epic100_2024-01-04_1601/frames/{}/rgb_frames/{}'.format(ann[1], ann[2]), 'frame_%010d'%_frame + '.jpg')
            assert os.path.exists(path)
            target_path = os.path.join(save_root, str(save_id), '{}.jpg'.format(index))
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(path, target_path)
            index += 1
        return False
    else:
        return True



if __name__ == '__main__':
    # load bbox annotations
    SAVE_IMAGES = True
    filename = "/shared_that_will_be_deleted_at_some_point/niudt/ORViT_BOX_H/whole_132028/epickitchens.h5" # only find it on fangtooth
    with h5py.File(filename, 'r') as f:
        bbox_annotations = f['hand_dets']

        # read train and val split of the epic-kitchens
        train_annotation_path = '/home/niudt/mae-cross-dev-r2d2/prepare_the_dataset/sample_videos/EPIC_100_train.csv'
        val_annotation_path = '/home/niudt/mae-cross-dev-r2d2/prepare_the_dataset/sample_videos/EPIC_100_validation.csv'
        stat = {}
        train_annotations = pd.read_csv(train_annotation_path).values
        val_annotations = pd.read_csv(val_annotation_path).values
        annotations = np.concatenate((train_annotations, val_annotations), axis=0)
        df = pd.read_csv(
            '/home/niudt/mae-cross-dev-r2d2/prepare_the_dataset/sample_videos/EPIC_100_verb_classes.csv').values

        for ann in tqdm(annotations):
            ann_id = ann[0]
            start_frame = ann[6]
            stop_frame = ann[7]
            length = stop_frame - start_frame
            verb_id = ann[10]
            verb = ann[9]

            assert df[verb_id][0] == verb_id
            verb = df[verb_id][1]
            instruction = ann[8]

            right_hands = []
            left_hands = []
            # here load the dets results
            for frame_idx in range(start_frame, stop_frame + 1):
                right_hand = bbox_annotations[ann[2]][str(frame_idx)][0]
                left_hand = bbox_annotations[ann[2]][str(frame_idx)][1]
                s = 1
                right_hands.append(right_hand)
                left_hands.append(left_hand)

            #TODO: BBOX: STYPLE
            #TODO: delete the fifth dimenstion bbox (confidence score)
            #TODO: how to transfer the bbox to trajectory
            #TODO: if all the bbox is 0, if w should delete this episode
            #TODO: how to save the trajectory joson in episode level

            s = 1

            # sample some examples
            if SAVE_IMAGES:
                save_root = '/scratch/partial_datasets/llarva/epic'
                save_root = os.path.join(save_root, verb)
                remove = save_episode(ann, ann_id, save_root)
                if remove:
                    continue









