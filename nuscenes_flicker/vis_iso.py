import numpy as np
import json
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--det_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--linear', type=int)

args = parser.parse_args()

with open(args.det_dir) as f:
    data = json.load(f)
    results = data['results']

plot_original = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
plot_original_positive = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plot_original_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

plot_iso = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
plot_iso_positive = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plot_iso_all = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for sample_token, objs in results.items():
    for obj in objs:
        bin_index = np.digitize(
            obj['detection_score'],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plot_original[bin_index - 1] += obj['detection_score']
        plot_original_positive[bin_index - 1] += obj['gt_score']
        plot_original_all[bin_index - 1] += 1
        bin_index = np.digitize(
            obj['iso_score'],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plot_iso[bin_index - 1] += obj['iso_score']
        plot_iso_positive[bin_index - 1] += obj['gt_score']
        plot_iso_all[bin_index - 1] += 1

fig = plt.figure(figsize=(10, 10))
plt.scatter(plot_original / (plot_original_all + 0.000001),
            plot_original_positive / (plot_original_all + 0.000001))
plt.plot([0, 1], [0, 1])
plt.xlabel('Original Score')
plt.ylabel('Positive Probability')
plt.savefig(osp.join(args.output_dir, 'original_{}.png'.format(args.linear)), bbox_inches='tight')
plt.close(fig)

fig = plt.figure(figsize=(10, 10))
plt.scatter(plot_iso / (plot_iso_all + 0.000001),
            plot_iso_positive / (plot_iso_all + 0.000001))
plt.plot([0, 1], [0, 1])
plt.xlabel('ISO Score')
plt.ylabel('Positive Probability')
plt.savefig(osp.join(args.output_dir, 'iso_{}.png'.format(args.linear)), bbox_inches='tight')
plt.close(fig)