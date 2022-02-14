import os
import os.path as osp
import numpy as np
import argparse
import json

# scene_names = [
#     'scenario_096', 'scenario_097', 'scenario_098', 'scenario_100',
#     'scenario_101', 'scenario_102', 'scenario_111', 'scenario_112',
#     'scenario_115', 'scenario_116', 'scenario_118', 'scenario_119',
#     'scenario_124', 'scenario_131', 'scenario_132', 'scenario_133',
#     'scenario_136', 'scenario_140', 'scenario_141', 'scenario_144',
# ]

# scene_names = [
#     'scenario_145', 'scenario_146', 'scenario_147', 'scenario_148',
#     'scenario_149', 'scenario_150', 'scenario_152', 'scenario_153',
#     'scenario_154', 'scenario_155'
# ]

# scene_names = [
#       'scenario_096', 'scenario_097', 'scenario_098', 'scenario_100',
#       'scenario_101', 'scenario_102', 'scenario_111', 'scenario_112',
#       'scenario_115', 'scenario_116', 'scenario_118', 'scenario_119',
#       'scenario_124', 'scenario_131', 'scenario_132', 'scenario_133',
#       'scenario_136', 'scenario_140', 'scenario_141', 'scenario_144',
#       'scenario_145', 'scenario_146', 'scenario_147', 'scenario_148',
#       'scenario_149', 'scenario_150', 'scenario_152', 'scenario_153',
#       'scenario_154', 'scenario_155'
# ]

threshold = {
    'Car': 0.5,
    'Pedestrian': 0.25,
    'Cyclist': 0.25,
    'Other vehicle': 0.5,
    'Bus': 0.5,
    'Truck': 0.5,
    'Motorcyclist': 0.25,
    'Animals': 0.25
}

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


def get_dets(det_dir, scene_name, class_index, threshold):
    det_file_dir = osp.join(det_dir, scene_name)
    file_names = os.listdir(det_file_dir)
    det_names = sorted(
        [file_name for file_name in file_names if 'detection' in file_name])
    detections = []
    for det_name in det_names:
        detection = {}
        det_file = osp.join(det_file_dir, det_name)
        det = np.loadtxt(det_file, delimiter=',', dtype='str')[1:, :]
        label_preds = det[:, 0]
        box3d_lidar = det[:, 1:8]
        ious = det[:, 8]
        scores = det[:, 9]
        cat_preds_tmp = [class_index[name] for name in label_preds]
        cat_preds = np.array(cat_preds_tmp).reshape(-1, )
        detection['box3d_lidar'] = box3d_lidar.astype(np.float32)
        detection['cat_preds'] = cat_preds
        detection['ious'] = ious.astype(np.float32)
        detection['scores'] = scores.astype(np.float32)
        labels_tmp = [
            float(iou) > threshold[name]
            for name, iou in zip(label_preds, ious)
        ]
        labels = np.array(labels_tmp).reshape(-1, )
        detection['labels'] = labels.astype(np.float32)
        detections.append(detection)
    return detections


def accumulator(detection):
    stats = {i: [[], []] for i in range(8)}
    for cat, label, score in zip(detection['cat_preds'], detection['labels'],
                                 detection['scores']):
        stats[cat][0].append(float(score))
        stats[cat][1].append(int(label))
    return stats


def get_scene_names(data_path):
    scene_names = []
    for file in sorted(os.listdir(data_path)):
        if 'scenario' in file:
            scene_names.append(file)
    return scene_names


def main(data_path, save_path, split='train'):
    # args = parse_args()
    scene_names = get_scene_names(data_path)

    all_stats = {i: [[], []] for i in range(8)}

    for scene_name in scene_names:
        detections = get_dets(data_path, scene_name, class_index, threshold)
        for detection in detections:
            stats = accumulator(detection)
            for key, value in stats.items():
                all_stats[key][0] += value[0]
                all_stats[key][1] += value[1]

    if not osp.isdir(osp.join(save_path)):
        os.makedirs(osp.join(save_path), exist_ok=True)
    if split == 'test':
        with open(osp.join(save_path, 'test_stats.json'), 'w') as outfile:
            json.dump(all_stats, outfile)
    else:
        with open(osp.join(save_path, 'train_stats.json'), 'w') as outfile:
            json.dump(all_stats, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Isotonic Regression")
    parser.add_argument('--det_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--split', type=str)
    args = parser.parse_args()
    main(args.det_dir, args.output_dir, args.split)

