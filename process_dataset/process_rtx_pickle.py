#!/usr/bin/env python

import argparse
import imgviz
import joblib
import os
import sys
from PIL import Image
import time


def vis_demo(path, demo_ind):
    """Visualizes a demo."""

    demo_dir = sorted(os.listdir(path))[demo_ind]
    demo_path = os.path.join(path, demo_dir)

    for fname in sorted(os.listdir(demo_path)):
        if not fname.endswith(".pkl"):
            continue
        with open(os.path.join(demo_path, fname), "rb") as f:
            data = joblib.load(f)
            print(list(data.keys()))
        im_tile = data["rgb_left"]

        image = Image.fromarray(im_tile)

        # Save the image
        save_root = './rpt_example'
        os.makedirs(save_root, exist_ok=True)
        image.save(os.path.join(save_root, '{}.jpg'.format(fname.split('.')[0])))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dir", dest="demo_dir", default="/home/niudt/dataset/rpt_demos") # this should be the outer path where saves all episode
    parser.add_argument("--demo-ind", dest="demo_ind", default=0, type=int)
    args = parser.parse_args()
    vis_demo(args.demo_dir, args.demo_ind)
