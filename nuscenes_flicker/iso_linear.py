import numpy as np
import os
import os.path as osp
import json
import argparse
from nuscenes import NuScenes
from .nuscenes_split import nuscenes_split
from pyquaternion import Quaternion
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--det_dir_iso', type=str)
parser.add_argument('--det_dir_unlabel', type=str)
parser.add_argument('--root_path', type=str, default='/working_dir/nuscenes')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--split', type=str, default='unlabeled')
parser.add_argument('--linear', type=int, default=1)

name_map = {
    'car': ['vehicle.car'],
    'pedestrian': [
        'human.pedestrian.adult', 'human.pedestrian.child',
        'human.pedestrian.construction_worker',
        'human.pedestrian.police_officer'
    ],
    'motorcycle': ['vehicle.motorcycle'],
    'bicycle': ['vehicle.bicycle'],
    'bus': ['vehicle.bus.bendy', 'vehicle.bus.rigid'],
    'truck': ['vehicle.truck'],
    'trailer': ['vehicle.trailer'],
    'construction_vehicle': ['vehicle.construction'],
    'barrier': ['movable_object.barrier'],
    'traffic_cone': ['movable_object.trafficcone']
}


def rescore(function, scores, linear):
    if linear == 1:
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_index = np.digitize(scores, bins=bins)
        current_score = function[bin_index - 1]
        if bin_index == 1:
            previous_score = 0
        else:
            previous_score = function[bin_index - 2]
        previous_bin = bins[bin_index - 1]
        new_score = (current_score - previous_score) / 0.1 * (
            scores - previous_bin) + previous_score
        return new_score
    else:
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_index = np.digitize(scores, bins=bins)
        return function[bin_index - 1]

def main(root_path, det_dir_iso, det_dir_unlabel, output_dir, split, linear):
    nusc_trainval = NuScenes(version='v1.0-trainval',
                             dataroot=root_path,
                             verbose=True)

    nusc_test = NuScenes(version='v1.0-test',
                         dataroot=root_path,
                         verbose=True)

    with open(det_dir_iso) as f:
        data = json.load(f)
        results_iso = data['results']

    with open(det_dir_unlabel) as f:
        data = json.load(f)
        results_unlabel = data['results']

    my_split = nuscenes_split('without_test', nusc_trainval, nusc_test)
    token_set_iso = my_split.get_all_sample_tokens('iso')
    if split == 'test':
        token_set_unlabel = my_split.get_all_sample_tokens('test')
    else:
        token_set_unlabel = my_split.get_all_sample_tokens('unlabeled_train')

    original_scores = []
    gt_scores = []
    for scene_name, token_time in token_set_iso.items():
        sample_tokens, timestamps = token_time[:]
        for sample_token, timestamp in zip(sample_tokens, timestamps):
            objs = results_iso[sample_token]
            for obj in objs:
                sample_info = nusc_trainval.get('sample', sample_token)
                anno_tokens = sample_info['anns']
                min_dis = 10
                original_scores.append(obj['detection_score'])
                for anno_token in anno_tokens:
                    gt_ann = nusc_trainval.get('sample_annotation', anno_token)
                    if gt_ann['category_name'] not in name_map[
                            obj['detection_name']]:
                        continue
                    cd = np.linalg.norm(
                        np.array(obj['translation'][:2]) -
                        np.array(gt_ann['translation'][:2]))
                    if cd < min_dis:
                        min_dis = cd
                if min_dis > 2.0:
                    gt_score = 0
                else:
                    gt_score = 1
                obj['gt_score'] = gt_score
                gt_scores.append(gt_score)

    bin_index = np.digitize(
        original_scores,
        bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    my_function = []
    my_function_all = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    my_function_positive = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, j in enumerate(bin_index - 1):
        my_function_all[j] += 1
        my_function_positive[j] += gt_scores[i]

    for i in range(10):
        my_function.append(my_function_positive[i] /
                           (my_function_all[i] + 0.00001))

    my_function_modified = []
    previous_value = 0
    for value in my_function:
        if value < previous_value:
            my_function_modified.append(previous_value + 0.001)
        else:
            my_function_modified.append(value)
        previous_value = my_function_modified[-1]
    my_function_modified = np.clip(my_function_modified, 0, 1.0 - 0.0001)
    print(my_function_modified)

    for scene_name, token_time in token_set_unlabel.items():
        sample_tokens, timestamps = token_time[:]
        for sample_token, timestamp in zip(sample_tokens, timestamps):
            objs = results_unlabel[sample_token]
            for obj in objs:
                iso_score = rescore(my_function_modified, obj['detection_score'], linear)
                obj['iso_score'] = iso_score

    with open(osp.join(output_dir, 'det_f_i_{}.json'.format(linear)),
              'w') as outfile:
        json.dump(data, outfile)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.root_path, args.det_dir_iso, args.det_dir_unlabel, args.output_dir, args.split, args.linear)