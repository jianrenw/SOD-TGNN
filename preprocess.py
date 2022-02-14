# @Author: hgang
# @Date:   2020-12-17
# @Last Modified by:   hgang
# @Last Modified time: 2020-12-17

#!/usr/bin/python
# @Author: Haiming Gang
# @Date:   2020-12-16

import os
import argparse
import CenterPoint_v1.tools.detection as centerpoint_detector
import CenterPoint_v1.tools.detection_batch as centerpoint_detector_batch
from det3d.datasets.h3d import h3d as h3d_ds
from label_generation import main as label_generation
from isotonic_regression import main as isotonic_regression
from vis_bin_new import main as vis_bin


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_detection(pc_path, ckpt_path, config, save_path, split='inference'):
    # centerpoint_detector.main(root_path=data_path, save_path=save_path, ckpt_path=ckpt_path)
    if split == 'val':
        centerpoint_detector_batch.main(
            ckpt_path=ckpt_path, config=config, save_path=save_path, split=split)
    else:
        if os.path.isfile('./h3d_all_infos_inference.pkl'):
            os.remove('./h3d_all_infos_inference.pkl')
        h3d_ds.create_h3d_inference_infos(pc_path, save_path='./')
    
        centerpoint_detector_batch.main(
            ckpt_path=ckpt_path, config=config, save_path=save_path)


def run_flicker(detection_path, save_path):
    try:
        command = (' ').join(
            ('./flicker.sh', detection_path, detection_path, save_path))
        # print(command)
        os.system(command)
    except ValueError:
        print('run flicker command wrong')


def run_isotonic_regression(gt_data_path, detection_path, isotonic_train_path):
    label_generation(gt_data_path, isotonic_train_path + 'inference/', isotonic_train_path + 'label_generation/')
    isotonic_regression(isotonic_train_path + 'label_generation/', isotonic_train_path)
    vis_bin(isotonic_train_path, detection_path, detection_path)


def hdd_data_prep(pc_path, label_path, save_path, gt_data_path=None):
    h3d_ds.create_h3d_infos(pc_path, label_path=label_path,
                            save_path=save_path, gt_data_path=gt_data_path, calib=False)
    # h3d_ds.create_reduced_point_cloud(data_path=data_path, save_path=data_path+"_reduced")
    # h3d_ds.create_groundtruth_database(save_path)
    # create_groundtruth_database("H3dDataset", data_path, Path(data_path) / "h3d_car_infos_train.pkl")


def h3d_test_prep(data_path, save_path=None):
    h3d_ds.create_h3d_test_infos(data_path, save_path=save_path, calib=False)
    # h3d_ds.create_reduced_point_cloud(data_path=data_path, save_path=data_path+"_reduced")
    # h3d_ds.create_groundtruth_database(save_path)
    # create_groundtruth_database("H3dDataset", data_path, Path(data_path) / "h3d_car_infos_train.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--root_path", required=True, help="root data path")
    parser.add_argument("--config", required=True,
                        help="train config file path")
    parser.add_argument("--round", type=int, required=True,
                        help="current round")
    # parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument(
        "--h3d_path", default='/working_dir/h3d_data/icra_bin/', help="h3d data path for test")
    parser.add_argument("--resume", type=bool, default=False,
                        help="the checkpoint file to resume from")
    parser.add_argument('--work_dir', type=str,
                        default='./CenterPoint_v1/work_dirs/round_')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    root_path = args.root_path
    h3d_path = args.h3d_path
    count = args.round

    data_path = root_path + '/HDD/'
    labels_path = root_path + '/labels/'
    if not os.path.isfile(h3d_path + 'h3d_all_infos_test.pkl'):
        h3d_test_prep(h3d_path)

    try:
        current_model_save = os.path.join(
            args.work_dir + str(count), args.config.split('/')[-1].split('.')[0], 'latest.pth')
        # save_path = './data_02/labels/round_' + str(count) + '/'
        save_path = labels_path + '/current_round/'
        save_path_val = save_path + 'isotonic_regression_train/'
        if not os.path.isfile(save_path + 'h3d_all_dbinfos_train.pkl'):
            create_folder(save_path)
            create_folder(save_path_val + 'inference/')
            run_detection(data_path, current_model_save,args.config, save_path)
            run_flicker(save_path, save_path)
            run_detection(data_path, current_model_save, args.config, save_path_val + 'inference/', split='val')
            run_flicker(save_path_val + 'inference/', save_path_val + 'inference/')
            run_isotonic_regression(h3d_path, save_path, save_path_val)
            hdd_data_prep(data_path, save_path, save_path, h3d_path)

    except ValueError:
        print('error!')


if __name__ == '__main__':
    main()

