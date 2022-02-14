import json
import numpy as np
import os
import os.path as osp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_det', type=str)
parser.add_argument('--val_det', type=str)
parser.add_argument('--test_det', type=int)
parser.add_argument('--output_dir', type=int)
args = parser.parse_args()

meta = {
    "use_camera": False,
    "use_lidar": True,
    "use_radar": False,
    "use_map": False,
    "use_external": False
}

merge_results = {}

for det_file in [args.train_det, args.val_det, args.test_det]:
    with open(det_file) as f:
        data = json.load(f)
        results = data['results']
        merge_results.update(results)

output_data = {'meta': meta, 'results': merge_results}
with open(osp.join(args.output_dir, 'det.json'), 'w') as outfile:
    json.dump(output_data, outfile)
