import numpy as np
import os
import os.path as osp
import json
import argparse
from nuscenes import NuScenes
from .nuscenes_split import nuscenes_split
import copy
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--det_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--root_path', type=str, default='/working_dir/nuscenes')
parser.add_argument('--split', type=str, default='unlabeled_train')
parser.add_argument('--frame', type=int, default=5)
parser.add_argument('--distance', type=int, default=10)
# parser.add_argument('--flag', type=str)

meta = {
    "use_camera": False,
    "use_lidar": True,
    "use_radar": False,
    "use_map": False,
    "use_external": False
}


def flicker(results, all_sample_tokens, all_timestamps, idx, my_obj, frame,
            distance):
    # search over frame
    # search over distance
    best_association_score = []
    start = max([0, idx - frame])
    end = min(idx + frame + 1, len(all_sample_tokens))
    for i in range(start, end):
        min_dis = distance
        local_best = 0
        if i == idx:
            continue
        time_lag = all_timestamps[i] - all_timestamps[idx]
        predict_position = time_lag / 1000000 * np.array(
            my_obj['velocity'][:2]) + np.array(my_obj['translation'][:2])
        if not all_sample_tokens[i] in results:
            import pdb; pdb.set_trace()
        objs = results[all_sample_tokens[i]]
        for obj in objs:
            if obj['detection_name'] != my_obj['detection_name']:
                continue
            cd = np.linalg.norm(
                np.array(obj['translation'][:2]) - predict_position)
            if cd < min_dis:
                min_dis = cd
                local_best = obj['detection_score']
        best_association_score.append(local_best)
    new_score = np.mean(np.array(best_association_score))
    return new_score * my_obj['detection_score']


def format_sample_result(sample_result, new_score):
    '''
    Output:
    sample_result {
        "sample_token":   <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
        "translation":    <float> [3]   -- Estimated bounding box location in meters in the global frame: center_x, center_y, center_z.
        "size":           <float> [3]   -- Estimated bounding box size in meters: width, length, height.
        "rotation":       <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
        "velocity":       <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
        "detection_name": predicted class for this sample_result, e.g. car, pedestrian.
                                        Note that the tracking_name cannot change throughout a track.
        "detection_score": <float>       -- Object prediction score between 0 and 1 for the class identified by tracking_name.
                                        We average over frame level scores to compute the track level score.
                                        The score is used to determine positive and negative tracks via thresholding.
        "original_score": <float> -- Original detection score.
        "attribute_name": str  -- Attribute_name
    }
    '''
    sample_result = copy.deepcopy(sample_result)
    sample_result["original_score"] = sample_result["detection_score"]
    sample_result["detection_score"] = new_score
    return sample_result

def main(root_path, det_dir, output_dir, split, frame, distance):
    nusc_trainval = NuScenes(version='v1.0-trainval',
                             dataroot=root_path,
                             verbose=True)

    nusc_test = NuScenes(version='v1.0-test',
                         dataroot=root_path,
                         verbose=True)

    my_split = nuscenes_split('without_test', nusc_trainval, nusc_test)
    token_set = my_split.get_all_sample_tokens(split)

    # with open(osp.join(args.det_dir, 'det.json')) as f:
    #     data = json.load(f)
    #     results = data['results']

    with open(det_dir) as f:
        data = json.load(f)
        results = data['results']

    new_results = {}
    # def save_flicker(single_token_set):
    #     scene_name, token_time = single_token_set
    #     # for scene_name, token_time in token_set.items():
    #     sample_tokens, timestamps = token_time[:]
    #     for idx, sample_token in enumerate(sample_tokens):
    #         # import pdb; pdb.set_trace()
    #         objects = results[sample_token]
    #         for obj in objects:
    #             new_score = flicker(results, sample_tokens, timestamps, idx, obj,
    #                                 args.frame, args.distance)
    #             if sample_token in new_results:
    #                 new_results[sample_token].append(
    #                     format_sample_result(obj, new_score))
    #             else:
    #                 new_results[sample_token] = [
    #                     format_sample_result(obj, new_score)
    #                 ]

    # pool = multiprocessing.Pool()
    # pool.map(save_flicker, token_set.items())
    for scene_name, token_time in token_set.items():
        sample_tokens, timestamps = token_time[:]
        for idx, sample_token in enumerate(sample_tokens):
            objects = results[sample_token]
            for obj in objects:
                new_score = flicker(results, sample_tokens, timestamps, idx, obj,
                                    frame, distance)
                if sample_token in new_results:
                    new_results[sample_token].append(
                        format_sample_result(obj, new_score))
                else:
                    new_results[sample_token] = [
                        format_sample_result(obj, new_score)
                    ]
    # print(len(new_results))

    if not osp.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_data = {'meta': meta, 'results': new_results}
    with open(osp.join(output_dir, 'det_{}.json'.format(split)), 'w') as outfile:
        json.dump(output_data, outfile)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.root_path, args.det_dir, args.output_dir, args.split, args.frame, args.distance)