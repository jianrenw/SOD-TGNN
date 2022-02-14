import numpy as np
import os
import os.path as osp
import json
from .utils import h3d
import argparse
import multiprocessing

global my_h3d
global mode
global output_dir

def get_adjacent_score_normal(dets, idx, my_obj):
    min_dis = 10
    best_association = np.array([1.0, 1.0])
    start = max([0, idx - 4])
    end = min(idx + 5, len(dets))
    for i in range(start, end):
        if i == idx:
            continue
        objs = dets[i]
        for obj in objs:
            if obj[0] != my_obj[0]:
                continue
            cd = np.linalg.norm(obj[5:7] - my_obj[5:7])
            if cd < min_dis:
                min_dis = cd
                best_association = np.array(obj[3:5])
    new_score = np.exp(-9 / 2 * (min_dis**2) /
                       (best_association[0] * best_association[1]))
    return new_score


def get_adjacent_score_high(dets, idx, my_obj):
    min_dis = 10
    best_association = np.array([1.0, 1.0])
    start = max([0, idx - 4])
    end = min(idx + 5, len(dets))
    for i in range(start, end):
        if i == idx:
            continue
        objs = dets[i]
        for obj in objs:
            if obj[0] != my_obj[0] or obj[1] < 0.5:
                continue
            cd = np.linalg.norm(obj[5:7] - my_obj[5:7])
            if cd < min_dis:
                min_dis = cd
                best_association = np.array(obj[3:5])
    new_score = np.exp(-9 / 2 * (min_dis**2) /
                       (best_association[0] * best_association[1]))
    return new_score


def get_adjacent_score_high_o(dets, idx, my_obj):
    min_dis = 10
    best_association = np.array([1.0, 1.0])
    start = max([0, idx - 4])
    end = min(idx + 5, len(dets))
    for i in range(start, end):
        if i == idx:
            continue
        objs = dets[i]
        for obj in objs:
            if obj[0] != my_obj[0] or obj[1] < 0.5:
                continue
            cd = np.linalg.norm(obj[5:7] - my_obj[5:7])
            if cd < min_dis:
                min_dis = cd
                best_association = np.array(obj[3:5])
    new_score = np.exp(-9 / 2 * (min_dis**2) /
                       (best_association[0] * best_association[1]))
    return new_score * my_obj[1]


def get_adjacent_score_hard(dets, idx, my_obj):
    best_association_score = []
    start = max([0, idx - 4])
    end = min(idx + 5, len(dets))
    for i in range(start, end):
        min_dis = 10
        local_best = 0
        if i == idx:
            continue
        objs = dets[i]
        for obj in objs:
            if obj[0] != my_obj[0]:
                continue
            cd = np.linalg.norm(obj[5:7] - my_obj[5:7])
            if cd < min_dis:
                min_dis = cd
                local_best = obj[1]
        best_association_score.append(local_best)
    if my_obj[1] < 0.3 and np.all(np.array(best_association_score) > 0.5):
        new_score = 0.9
    elif my_obj[1] > 0.7 and np.all(np.array(best_association_score) < 0.5):
        new_score = 0.1
    else:
        return my_obj[1]
    return new_score


def get_adjacent_score_average(dets, idx, my_obj):
    best_association_score = []
    start = max([0, idx - 4])
    end = min(idx + 5, len(dets))
    for i in range(start, end):
        min_dis = 10
        local_best = 0
        if i == idx:
            continue
        objs = dets[i]
        for obj in objs:
            if obj[0] != my_obj[0]:
                continue
            cd = np.linalg.norm(obj[5:7] - my_obj[5:7])
            if cd < min_dis:
                min_dis = cd
                local_best = obj[1]
        best_association_score.append(local_best)
    new_score = np.mean(np.array(best_association_score))
    return new_score


def get_adjacent_score_average_o(dets, idx, my_obj):
    best_association_score = []
    start = max([0, idx - 4])
    end = min(idx + 5, len(dets))
    for i in range(start, end):
        min_dis = 10
        local_best = 0
        if i == idx:
            continue
        objs = dets[i]
        for obj in objs:
            if obj[0] != my_obj[0]:
                continue
            cd = np.linalg.norm(obj[5:7] - my_obj[5:7])
            if cd < min_dis:
                min_dis = cd
                local_best = obj[1]
        best_association_score.append(local_best)
    new_score = np.mean(np.array(best_association_score))
    return new_score * my_obj[1]


def format_sample_result(original_det, new_score):
    str = '%s,%s,%s,%s,%s,%s,%s,%s,%f,%f,%s \n' % (
        original_det[0], original_det[1], original_det[2], original_det[3],
        original_det[4], original_det[5], original_det[6], original_det[7],
        new_score, new_score, original_det[10])
    return str


def save_flicker(scene_name, score_threshold=0.1):
    # if not osp.isdir(
    #         osp.join(args.output_dir, args.projection, args.mode, scene_name)):
    #     os.makedirs(osp.join(args.output_dir, args.projection, args.mode,
    #                          scene_name),
    #                 exist_ok=True)
    global my_h3d
    global mode
    global output_dir

    if not osp.isdir(
            osp.join(output_dir, scene_name)):
        os.makedirs(osp.join(output_dir, scene_name),
                    exist_ok=True)
    dets = my_h3d.get_dets(scene_name)
    original_dets = my_h3d.get_original_dets(scene_name)
    for idx, (det, original_det) in enumerate(zip(dets, original_dets)):
        # save_file_name = os.path.join(args.output_dir, args.projection,
        #                               args.mode, scene_name,
        #                               'labels_3d1_{:03d}.txt'.format(idx))
        save_file_name = os.path.join(output_dir, scene_name,
                                      'labels_3d1_{:03d}.txt'.format(idx))

        save_file = open(save_file_name, 'w')
        str = 'class,x,y,z,w(l_x),l(l_y),h(l_z),yaw,score,probability,num_points \n'
        save_file.write(str)
        for i, my_obj in enumerate(det):
            if mode == 'normal':
                new_score = get_adjacent_score_normal(dets, idx, my_obj)
            elif mode == 'high':
                new_score = get_adjacent_score_high(dets, idx, my_obj)
            elif mode == 'high_o':
                new_score = get_adjacent_score_high_o(dets, idx, my_obj)
            elif mode == 'hard':
                new_score = get_adjacent_score_hard(dets, idx, my_obj)
            elif mode == 'average':
                new_score = get_adjacent_score_average(dets, idx, my_obj)
            elif mode == 'average_o':
                new_score = get_adjacent_score_average_o(dets, idx, my_obj)

            if new_score >= score_threshold: 
                str = format_sample_result(original_det[i + 1], new_score)
                save_file.write(str)
        save_file.close()

def main(gt_dir, det_dir, flicker_output_dir, projection='original', flicker_mode='average_o'):
    # if not osp.isdir(osp.join(args.output_dir, args.projection, args.mode)):
    #     os.makedirs(osp.join(args.output_dir, args.projection, args.mode),
    #                 exist_ok=True)

    global my_h3d
    global mode
    global output_dir
    my_h3d = h3d(gt_dir, det_dir, projection)
    mode = flicker_mode
    output_dir = flicker_output_dir
    # scene_names = [
    #     'scenario_096', 'scenario_097', 'scenario_098', 'scenario_100',
    #     'scenario_101', 'scenario_102', 'scenario_111', 'scenario_112',
    #     'scenario_115', 'scenario_116', 'scenario_118', 'scenario_119',
    #     'scenario_124', 'scenario_131', 'scenario_132', 'scenario_133',
    #     'scenario_136', 'scenario_140', 'scenario_141', 'scenario_144',
    #     'scenario_145', 'scenario_146', 'scenario_147', 'scenario_148',
    #     'scenario_149', 'scenario_150', 'scenario_152', 'scenario_153',
    #     'scenario_154', 'scenario_155'
    # ]

    scene_names = []
    for folder in sorted(os.listdir(det_dir)):
        if 'scenario' in folder:
            scene_names.append(folder)

    pool = multiprocessing.Pool()
    pool.map(save_flicker, scene_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--det_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--projection', type=str)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    main(args.gt_dir, args.det_dir, args.output_dir, args.projection, args.mode)