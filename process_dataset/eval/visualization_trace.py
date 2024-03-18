import json
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from PIL import Image
import os
import sys


def plot_points_on_image(image_path, words_and_coords_pred, words_and_coords_gt, output_folder):
    img = Image.open(os.path.join('/home/niudt/dataset/franctal_llarva', image_path))
    plt.imshow(img)
    print(words_and_coords_pred)

    # Plot points
    prev_coord = None
    for word, coord in words_and_coords_gt.items():
        x, y, shape = coord
        if shape == 0:
            continue
        marker = 'o' if shape == 0 else '*'
        plt.scatter(x, y, marker=marker, label=word, color="green")
        if prev_coord is not None:
            print("prev coord is: ", prev_coord)
            prev_x, prev_y, shape2 = prev_coord
            plt.plot([prev_x, x], [prev_y, y], color='green')
        prev_coord = (x, y, shape)

    prev_coord = None

    for word, coord in words_and_coords_pred.items():
        x, y, shape = coord
        if shape == 0:
            continue
        marker = 'o' if shape == 0 else '*'
        plt.scatter(x, y, marker=marker, label=word, color="red")

        # Add text annotation near the point
        # plt.text(x, y, word, fontsize=8, ha='right', va='bottom')
        if prev_coord is not None:
            print("prev coord is: ", prev_coord)
            prev_x, prev_y, shape2 = prev_coord
            plt.plot([prev_x, x], [prev_y, y], color='red')
        prev_coord = (x, y, shape)





    plt.title("pick green jalapeno chip bag from bottom drawer and place on counter")
    # plt.legend()
    # Save the edited image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path)
    plt.close()


# Load JSONL file and process each line
jsonl_file = '/home/niudt/LLaVA/process_dataset/eval/demo_pixel_answers.jsonl'
output_folder = '/home/niudt/LLaVA/process_dataset/eval/visualizations/demo_trace'
os.makedirs(output_folder, exist_ok=True)

with open(jsonl_file, 'r') as f:
    for line in f:
        data = json.loads(line)

        image_path = data['image_path']
        text = data['text']
        words_and_coords_pred = {}
        length = len(text.split(']'))
        counter = 0
        for word_info in text.split(']'):
            if counter == length - 1:
                continue
            word = word_info.strip().split(' ')[0]

            coords_string = word_info.split('[')[1].replace(' ', '')
            coords = map(int, coords_string.split(','))
            words_and_coords_pred[word] = coords
            counter += 1

        gt = data['gt']
        words_and_coords_gt = {}
        length = len(gt.split(']'))
        counter = 0
        for word_info in gt.split(']'):
            if counter == length - 1:
                continue
            word = word_info.strip().split(' ')[0]
            coords_string = word_info.split('[')[1].replace(' ', '')
            coords = map(int, coords_string.split(','))
            words_and_coords_gt[word] = coords
            counter += 1

        plot_points_on_image(image_path, words_and_coords_pred, words_and_coords_gt, output_folder)
