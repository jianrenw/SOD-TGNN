#!/usr/bin/python
# @Author: Haiming Gang
# @Date:   2021-04-07
import numpy as np
import os
import os.path as osp
import json
import argparse
from pyquaternion import Quaternion
import copy

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")

parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--root_path', type=str, default='/working_dir/nuscenes')

def main(root_path, json_file, output_dir):
	nusc_train = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)
	nusc_test = NuScenes(version='v1.0-test', dataroot=root_path, verbose=True)

	with open(json_file) as f:
	    data = json.load(f)
	    results_unlabel = data['results']

	for sample_token in results_unlabel:
		try:
			s_record = nusc_train.get("sample", sample_token)
			sample_data_token = s_record["data"]["LIDAR_TOP"]
			sd_record = nusc_train.get("sample_data", sample_data_token)
			cs_record = nusc_train.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
			pose_record = nusc_train.get("ego_pose", sd_record["ego_pose_token"])
		except:
			s_record = nusc_test.get("sample", sample_token)
			sample_data_token = s_record["data"]["LIDAR_TOP"]
			sd_record = nusc_test.get("sample_data", sample_data_token)
			cs_record = nusc_test.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
			pose_record = nusc_test.get("ego_pose", sd_record["ego_pose_token"])

		for i in range(len(results_unlabel[sample_token])):
			box3d = results_unlabel[sample_token][i]

			box = Box(
				box3d['translation'],
			    box3d['size'],
			    Quaternion(box3d['rotation']),
			    # label=box3d['detection_name'],
			    # score=box3d['detection_score'],
			    velocity = (box3d['velocity'][0], box3d['velocity'][1], 0.0),
			 )

			box.translate(-np.array(pose_record["translation"]))
			box.rotate(Quaternion(pose_record["rotation"]).inverse)

			#  Move box to sensor coord system
			box.translate(-np.array(cs_record["translation"]))
			box.rotate(Quaternion(cs_record["rotation"]).inverse)

			results_unlabel[sample_token][i]["translation"] = box.center.tolist()
			results_unlabel[sample_token][i]["size"] = box.wlh.tolist()
			results_unlabel[sample_token][i]["rotation"] = box.orientation.elements.tolist()
			results_unlabel[sample_token][i]["velocity"] = box.velocity[:2].tolist()

	with open(osp.join(output_dir, 'det_local.json'), 'w') as outfile:
	    json.dump(data, outfile)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args.root_path, args.json_file, args.output_dir)