import itertools
import logging

# from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=8, class_names=["Car", "Pedestrian", "Other vehicle", "Truck", "Bus", "Motorcyclist", "Cyclist", "Animals"])
    # dict(num_class=1, class_names=["Car"]),
    # dict(num_class=1, class_names=["Pedestrian"]),
    # dict(num_class=1, class_names=["Other vehicle"]),
    # dict(num_class=2, class_names=["Truck", "Bus"]),
    # dict(num_class=2, class_names=["Motorcyclist", "Cyclist"]),
    # dict(num_class=1, class_names=["Animals"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        # type='SimpleVoxel',
        num_input_features=4,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=4, ds_factor=8,
    ),
    # backbone=dict(
    #     type="SpMiddleFHD", num_input_features=4, ds_factor=8, norm_cfg=norm_cfg,
    # ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='h3d',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)},
        semi_weight=0.5,
        semi_thresh=0.5,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

# test_cfg = dict(
#     post_center_limit_range=[-1000., -1000., -1000.0, 1000., 1000.0, 1000.0],
#     max_per_img=500,
#     nms=dict(
#         use_rotate_nms=True,
#         use_multi_class_nms=False,
#         nms_pre_max_size=4096,
#         nms_post_max_size=500,
#         nms_iou_threshold=0.7,
#     ),
#     score_threshold=0.1,
#     pc_range=[-40, -40],
#     out_size_factor=get_downsample_factor(model),
#     # voxel_size=[0.075, 0.075],
#     voxel_size=[0.05, 0.05],
# )
test_cfg = dict(
    post_center_limit_range=[-1000., -1000., -1000.0, 1000., 1000.0, 1000.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.1,
    ),
    score_threshold=0.1,
    pc_range=[-40, -40],
    out_size_factor=get_downsample_factor(model),
    # voxel_size=[0.075, 0.075],
    voxel_size=[0.05, 0.05],
)

# dataset settings
dataset_type = "H3dDataset"
nsweeps = 10
data_root = "/working_dir/h3d_data/icra_bin"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="/working_dir/h3d_data/icra_bin/h3d_all_dbinfos_train.pkl",
    sample_groups=[
        dict(Car=2),
        dict(Pedestrian=2),
        {'Other vehicle':6},
        dict(Truck=3),
        dict(Bus=4),
        dict(Motorcyclist=6),
        dict(Cyclist=6),
        # dict(Other vehicle=6),
        dict(Animals=2),
    ],
    db_prep_steps=[
        dict(
            # filter_by_min_num_points=dict(
            #     Car=5,
            #     Pedestrian=5,
            #     Cyclist=5,
            #     # Other vehicle = 5,
            #     # Other vehicle=5,
            #     Bus=5,
            #     Truck=5,
            #     Motorcyclist=5,
            #     Animals=5,
            # )
            filter_by_min_num_points={
                'Car':5,
                'Pedestrian':5,
                'Other vehicle':5,
                'Truck':5,
                'Bus':5,
                'Motorcyclist':5,
                'Cyclist':5,
                'Animals':5,
            }
        ),
        # dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[1.0, 1.0, 0.5],
    gt_rot_noise=[-0.785, 0.785],
    global_rot_noise=[-0.785, 0.785],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.0, 0.0, 0.0],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,

    # mode="train",
    # shuffle_points=True,
    # gt_loc_noise=[0.0, 0.0, 0.0],
    # gt_rot_noise=[0.0, 0.0],
    # global_rot_noise=[-0.78539816, 0.78539816],
    # global_scale_noise=[0.95, 1.05],
    # global_rot_per_obj_range=[0, 0],
    # global_trans_noise=[0, 0, 0],
    # remove_points_after_sample=False,
    # gt_drop_percentage=0.0,
    # gt_drop_max_keep_points=15,
    # remove_unknown_examples=False,
    # remove_environment=False,
    # db_sampler=db_sampler,
    # class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[-40, -40, -3, 40, 40, 1],
    voxel_size=[0.05, 0.05, 0.1],
    max_points_in_voxel=5,
    max_voxel_num=80000,
    # range=[-40, -40, -3, 40, 40, 1],
    # voxel_size=[0.075, 0.075, 0.2],
    # max_points_in_voxel=10,
    # max_voxel_num=90000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = "/working_dir/h3d_data/icra_bin/h3d_all_infos_train.pkl"
val_anno = "/working_dir/h3d_data/icra_bin/h3d_all_infos_val.pkl"
test_anno = "/working_dir/h3d_data/icra_bin/h3d_all_infos_test.pkl"
inf_anno = "/working_dir/semi_supervised/data/HDD/h3d_all_infos_inference.pkl"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    inference=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=inf_anno,
        ann_file=inf_anno,
        test_mode=True,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 1
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
# workflow = [('train', 1)]
workflow = [("train", 1), ("val", 1)]
