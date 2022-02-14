import numpy as np
import json
import copy
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt

class_index = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Other vehicle': 3,
    'Bus': 4,
    'Truck': 5,
    'Motorcyclist': 6,
    'Animals': 7
}

def get_scene_names(data_path):
    scene_names = []
    for file in sorted(os.listdir(data_path)):
        if 'scenario' in file:
            scene_names.append(os.path.join(data_path, file))
    return scene_names

def rescore(function, scores, cat_index):
    bin_index = np.digitize(
        scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tmp_function = np.array(function[cat_index])
    new_score = tmp_function[bin_index - 1]
    return new_score

def main(train_data_path, val_data_path, save_path):
    with open(osp.join(train_data_path, 'train_stats.json')) as f:
        val_train_stats = json.load(f)
    
    if not osp.isdir(osp.join(save_path)):
        os.makedirs(osp.join(save_path), exist_ok=True)
    
    categories = [
        'car', 'pedestrian', 'cyclist', 'other vehicle', 'bus', 'truck',
        'motorcyclist', 'animals'
    ]
    
    
    train_bin_function_positive = {
        name: [0 for i in range(10)]
        for name in range(8)
    }
    train_bin_function_score = {name: [0 for i in range(10)] for name in range(8)}
    train_bin_function_all = {name: [0 for i in range(10)] for name in range(8)}
    train_bin_function_final = {name: [0 for i in range(10)] for name in range(8)} # This is what we want
    train_bin_function_b = {name: [0 for i in range(10)] for name in range(8)}
    
    # # new_score = train_bin_function_final[class][bin_index]
    
    for key, value in val_train_stats.items():
        scores = value[0]
        labels = value[1]
        bin_index = np.digitize(
            scores, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        for i, j in enumerate(bin_index - 1):
            train_bin_function_all[int(key)][j] += 1
            train_bin_function_score[int(key)][j] += scores[i]
            train_bin_function_positive[int(key)][j] += labels[i]
    
    for i in range(8):
        for j in range(10):
            train_bin_function_final[i][j] = train_bin_function_positive[i][j] / (
                train_bin_function_all[i][j] + 0.00001)
            train_bin_function_b[i][j] = train_bin_function_score[i][j] / (
                train_bin_function_all[i][j] + 0.00001)
    
    scene_names = get_scene_names(val_data_path)
    for folder in scene_names:
        for file in sorted(os.listdir(folder)):
            if 'flicker' in file:
                text_file = os.path.join(folder, file)
                if not os.path.exists(folder.replace(val_data_path,save_path)):
                    os.makedirs(folder.replace(val_data_path,save_path))
                text_save = os.path.join(folder.replace(val_data_path,save_path), file.replace('flicker', 'labels_3d1'))
                flick_file = os.path.join(folder, file)
                raw_data = np.loadtxt(text_file, delimiter=',', dtype='str')
                tile = raw_data[0,:]
                data = raw_data[1:,:]
                labels = data[:, 0]
                scores = copy.deepcopy(data[:, 8].astype(np.float32))
                probes = copy.deepcopy(data[:, 9].astype(np.float32))
                for i in range(len(scores)):
                    new_scores = rescore(train_bin_function_final, scores[i], class_index[labels[i]])
                    new_probes = rescore(train_bin_function_final, probes[i], class_index[labels[i]])
                    data[i][8] = str(new_scores)
                    data[i][9] = str(new_probes)

                save_data =  np.vstack((tile, data))
                # print(text_save)
                np.savetxt(text_save, save_data, delimiter=',', fmt='%s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--val_data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    main(args.train_data_path, args.val_data_path, args.output_dir)