#!/usr/bin/python
# @Author: Haiming Gang
# @Date:   2020-12-16

import os
import argparse
from pathlib import Path
import tools.detection_batch as centerpoint_detector_batch
from det3d.datasets.h3d import h3d as h3d_ds
from h3d_flicker.flicker import main as flicker
from h3d_flicker.label_generation import main as label_generation
from h3d_flicker.isotonic_regression import main as isotonic_regression
from h3d_flicker.vis_bin_new import main as vis_bin
import json
import sys

from datetime import datetime
import numpy as np
import torch
import yaml
from det3d.datasets import build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_detection(pc_path, ckpt_path, config, save_path, split='inference'):
    # centerpoint_detector.main(root_path=data_path, save_path=save_path, ckpt_path=ckpt_path)
    if split == 'val':
        centerpoint_detector_batch.main(
            ckpt_path=ckpt_path, config=config, save_path=save_path, split=split)
    else:
        # if os.path.isfile('./h3d_all_infos_inference.pkl'):
        #     os.remove('./h3d_all_infos_inference.pkl')
        # h3d_ds.create_h3d_inference_infos(pc_path, save_path='./')
        centerpoint_detector_batch.main(
            ckpt_path=ckpt_path, config=config, save_path=save_path)


def run_flicker(det_dir, save_path, projection='original', flicker_mode='average_o'):
    flicker(det_dir, det_dir, save_path, projection='original', flicker_mode='average_o')

def run_iso(gt_data_path, detection_path, isotonic_train_path):
    label_generation(gt_data_path, isotonic_train_path + 'inference/', isotonic_train_path + 'label_generation/')
    isotonic_regression(isotonic_train_path + 'label_generation/', isotonic_train_path)
    vis_bin(isotonic_train_path, detection_path, detection_path)


def hdd_data_prep(pc_path, label_path, save_path, data_len, gt_data_path=None):
    h3d_ds.create_h3d_infos(pc_path, label_path=label_path,
                            save_path=save_path, gt_data_path=gt_data_path, calib=False, semi=True, data_len=data_len)
    # h3d_ds.create_reduced_point_cloud(data_path=data_path, save_path=data_path+"_reduced")
    h3d_ds.create_groundtruth_database(save_path)
    # create_groundtruth_database("H3dDataset", data_path, Path(data_path) / "h3d_car_infos_train.pkl")


def h3d_test_prep(data_path, save_path=None):
    h3d_ds.create_h3d_test_infos(data_path, save_path=save_path, calib=False)


def run_train(args, train_anno=None):
    # args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    # if args.resume_from is not None:
    #     cfg.resume_from = args.resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from

    if train_anno is not None:
        cfg.train_anno = train_anno
        cfg.data.train.info_path = train_anno
        cfg.data.train.ann_file = train_anno

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()

    if args.autoscale_lr:
        cfg.lr_config.lr_max = cfg.lr_config.lr_max * cfg.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed training: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    if args.local_rank == 0:
        # copy important files to backup
        print(cfg.work_dir)
        backup_dir = os.path.join(cfg.work_dir, "det3d")
        os.makedirs(backup_dir, exist_ok=True)
        # os.system("cp -r * %s/" % backup_dir)
        # logger.info(f"Backup source files to {cfg.work_dir}/det3d")

    # set random seeds
    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        set_random_seed(args.seed)

    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    if cfg.checkpoint_config is not None:
        # save det3d version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            config=cfg.text, CLASSES=datasets[0].CLASSES
        )

    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        logger=logger,
    )


def move_next_round(label_path, count):
    command = (' ').join(('mv', label_path + 'current_round',
                          label_path + '/round_' + str(count)))
    # print(command)
    os.system(command)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    # parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument("--save_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    # parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--load_from", help="the checkpoint file to resume from")
    # parser.add_argument("--resume", type=bool, default=False,
    #                     help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--total_rounds", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument('--root_path', type=str, default='/working_dir/nuscenes')
    # parser.add_argument("--root_path", type=str,
    #                     default='/working_dir/semi_supervised/data_center_v1')
    parser.add_argument("--data_path", type=str,
                        default='/working_dir/semi_supervised/data/HDD/')
    parser.add_argument("--h3d_data", type=str,
                        default='/working_dir/h3d_data/icra_bin/')
    parser.add_argument('--work_dir', type=str,
                        default='./CenterPoint_v1/work_dirs/round_')
    parser.add_argument('--train_pickle', type=str, 
                        default='/working_dir/nuscenes/infos_train_10sweeps_withvelo_filter_True.pkl')
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    args = parser.parse_args()

    return args


def main(args):
    if not os.path.isfile(os.path.join(args.data_path, './h3d_all_infos_inference.pkl')):
        h3d_ds.create_h3d_inference_infos(args.data_path, save_path=args.data_path)

    # root_path = args.root_path
    date = datetime.today().strftime("%Y%m%d_%H%M%S")
    config_name = args.config

    config_name = (
        config_name.split("/")[-1].split(".")[0]
        + "_"
        + date
    )
    save_path = os.path.join(args.save_dir, config_name)
    # save_path = args.save_dir

    for round in range(1, args.total_rounds + 1):
        if round == 1:
            checkpoint = args.checkpoint
        else:
            checkpoint = os.path.join(save_path, 'round_' + str(round - 1).zfill(2), 'model', 'latest.pth')
        args.checkpoint = checkpoint
        args.load_from = checkpoint
        data_len = round * 4. /  args.total_rounds
        data_len = min(data_len, 1.)
        # print(data_len)
        current_save_path = os.path.join(save_path, 'round_' + str(round).zfill(2), 'data')
        current_model_path = os.path.join(save_path, 'round_' + str(round).zfill(2), 'model')
        if not os.path.isdir(current_save_path):
            os.makedirs(current_save_path, exist_ok=True)
        if not os.path.isdir(current_model_path):
            os.makedirs(current_model_path, exist_ok=True)
        if not os.path.exists(os.path.join(current_save_path, 'h3d_all_infos_train.pkl')):
            run_detection(args.data_path, checkpoint, args.config, current_save_path)
            # run_flicker(current_save_path, current_save_path)
            # run_detection(args.data_path, checkpoint, args.config, os.path.join(current_save_path, 'inference/'), split='val')
            # run_flicker(current_save_path + 'inference/', current_save_path + 'inference/')
            # run_iso(args.h3d_data, save_path, current_save_path)
            hdd_data_prep(args.data_path, current_save_path, current_save_path, data_len, args.h3d_data)
        
        args.work_dir = current_model_path
        train_anno = os.path.join(current_save_path, 'h3d_all_infos_train.pkl')
        run_train(args, train_anno)
        # # break


if __name__ == '__main__':
    main(parse_args())

