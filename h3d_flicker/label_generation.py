import os
import os.path as osp
import numpy as np
import sys
import numba
from det3d.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval
# from second.core.non_max_suppression.nms_gpu import rotate_iou_gpu_eval
import argparse

det_class_index = {
    'Car': 0,
    'Pedestrian': 1,
    'Cyclist': 2,
    'Other vehicle': 3,
    'Bus': 4,
    'Truck': 5,
    'Motorcyclist': 6,
    'Animals': 7
}

gt_class_index = {
    'car': 0,
    'pedestrian': 1,
    'cyclist': 2,
    'other vehicle': 3,
    'bus': 4,
    'truck': 5,
    'motorcyclist': 6,
    'animals': 7
}


def get_start_det_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations


def get_empty_det_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations


def get_start_gt_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
    })
    return annotations


def get_empty_gt_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
    })
    return annotations


def get_original_dets(det_dir, scene_name):
    dets = []
    file_dir = osp.join(det_dir, scene_name)
    file_names = os.listdir(file_dir)
    det_names = sorted(
        [file_name for file_name in file_names if 'flicker' in file_name])
    for det_name in det_names:
        det_file = osp.join(file_dir, det_name)
        det = np.loadtxt(det_file, delimiter=',', dtype='str')[1:, :]
        dets.append(det)
    return dets


def get_dets(det_dir, scene_name, class_index):
    det_file_dir = osp.join(det_dir, scene_name)
    file_names = os.listdir(det_file_dir)
    det_names = sorted(
        [file_name for file_name in file_names if 'flicker' in file_name])
    detections = []
    for det_name in det_names:
        detection = {
            "box3d_lidar": None,
            'label_preds': None,
            'scores': None,
        }
        det_file = osp.join(det_file_dir, det_name)
        det = np.loadtxt(det_file, delimiter=',', dtype='str')[1:, :]
        label_preds = det[:, 0]
        box3d_lidar = det[:, 1:8]
        scores = det[:, 8]
        label_preds_tmp = [class_index[name] for name in label_preds]
        label_preds = np.array(label_preds_tmp).reshape(-1, )
        detection['box3d_lidar'] = box3d_lidar.astype(np.float32)
        detection['label_preds'] = label_preds
        detection['scores'] = scores.astype(np.float32)
        detections.append(detection)
    return detections


def get_gts(gt_dir, scene_name, class_index):
    file_dir = osp.join(gt_dir, scene_name)
    file_names = os.listdir(file_dir)
    label_names = sorted(
        [file_name for file_name in file_names if 'labels' in file_name])
    gts = []
    for label_name in label_names:
        gt = {'box3d_lidar': None, 'label_preds': None}
        label_file = osp.join(file_dir, label_name)
        label = np.loadtxt(label_file, delimiter=',', dtype='str')
        category = label[:, 0]
        box3d_lidar = label[:, 3:10]
        label_preds_tmp = [class_index[name] for name in category]
        category = np.array(label_preds_tmp).reshape(-1, )
        gt['box3d_lidar'] = box3d_lidar.astype(np.float32)
        gt['label_preds'] = category
        gts.append(gt)
    return gts


def convert_detection_to_h3d_annos(detections):
    annos = []
    for i in range(len(detections)):
        det = detections[i]
        final_box_preds = det["box3d_lidar"]
        label_preds = det["label_preds"]
        scores = det["scores"]

        anno = get_start_det_anno()
        box3d_lidar = final_box_preds
        num_example = 0
        # limit_range = np.array([0, -40, -2.2, 70.4, 40, 0.8])
        for j in range(box3d_lidar.shape[0]):
            # convert center format to kitti format
            anno["bbox"].append(np.array([0, 0, 50, 50]))
            anno["alpha"].append(-10)
            anno["dimensions"].append(box3d_lidar[j, 3:6])
            anno["location"].append(box3d_lidar[j, :3])
            anno["rotation_y"].append(box3d_lidar[j, 6])
            anno["name"].append(label_preds)
            anno["truncated"].append(0.0)
            anno["occluded"].append(0)
            anno["score"].append(scores[j])
            num_example += 1
        if num_example != 0:
            anno = {n: np.stack(v) for n, v in anno.items()}
            annos.append(anno)
        else:
            annos.append(get_empty_det_anno())
    return annos


def convert_gt_to_h3d_annos(gts):
    annos = []
    for i in range(len(gts)):
        det = gts[i]
        final_box = det["box3d_lidar"]
        label = det["label_preds"]

        anno = get_start_gt_anno()
        box3d_lidar = final_box
        num_example = 0
        # limit_range = np.array([0, -40, -2.2, 70.4, 40, 0.8])
        for j in range(box3d_lidar.shape[0]):
            # convert center format to kitti format
            anno["bbox"].append(np.array([0, 0, 50, 50]))
            anno["alpha"].append(-10)
            anno["dimensions"].append(box3d_lidar[j, 3:6])
            anno["location"].append(box3d_lidar[j, :3])
            anno["rotation_y"].append(box3d_lidar[j, 6])
            anno["name"].append(label)
            anno["truncated"].append(0.0)
            anno["occluded"].append(0)
            num_example += 1
        if num_example != 0:
            anno = {n: np.stack(v) for n, v in anno.items()}
            annos.append(anno)
        else:
            annos.append(get_empty_gt_anno())
    return annos


@numba.jit(nopython=True, parallel=True)
def box3d_overlap_kernel(boxes,
                         qboxes,
                         rinc,
                         criterion=-1,
                         z_axis=1,
                         z_center=1.0):
    """
        z_axis: the z (height) axis.
        z_center: unified z (height) center of box.
    """
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center))
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center)
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def box3d_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)

    # t = time.time()
    # rinc = box_np_ops.rinter_cc(boxes[:, bev_axes], qboxes[:, bev_axes])
    rinc = rotate_iou_gpu_eval(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    # print("riou time", time.time() - t)
    box3d_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


def cal_iou(gts, detections, z_axis, z_center):
    overlaps = []
    for gt, detection in zip(gts, detections):
        loc = gt["location"]
        dims = gt["dimensions"]
        rots = gt["rotation_y"]
        gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        loc = detection["location"]
        dims = detection["dimensions"]
        rots = detection["rotation_y"]
        dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        overlap = box3d_overlap(gt_boxes,
                                dt_boxes,
                                z_axis=z_axis,
                                z_center=z_center).astype(np.float64)
        gt_name_matrix = np.tile(gt["name"][:1, :].transpose(),
                                 (1, detection["name"].shape[0]))
        det_name_matrix = np.tile(detection["name"][:1, :],
                                  (gt["name"].shape[0], 1))
        mask = np.equal(gt_name_matrix, det_name_matrix)
        overlap_per_cat = np.where(mask, overlap, 0)
        overlaps.append(overlap_per_cat)
    return overlaps


def format_sample_result(original_det, new_score):
    str = '%s,%s,%s,%s,%s,%s,%s,%s,%f,%s,%s \n' % (
        original_det[0], original_det[1], original_det[2], original_det[3],
        original_det[4], original_det[5], original_det[6], original_det[7],
        new_score, original_det[8], original_det[10])
    return str

def get_scene_names(data_path):
    scene_names = []
    for file in sorted(os.listdir(data_path)):
        if 'scenario' in file:
            scene_names.append(file)
    return scene_names

def main(gt_dir, det_dir, output_dir):
    # scene_names = os.listdir(args.det_dir)
    scene_names = get_scene_names(det_dir)
    
    for scene_name in scene_names:
        detections = get_dets(det_dir, scene_name, det_class_index)
        gts = get_gts(gt_dir, scene_name, gt_class_index)
        detections = convert_detection_to_h3d_annos(detections)
        gts = convert_gt_to_h3d_annos(gts)
        overlaps = cal_iou(gts, detections, z_axis=1, z_center=1.0)
        original_dets = get_original_dets(det_dir, scene_name)
    
        if not osp.isdir(osp.join(output_dir, scene_name)):
            os.makedirs(osp.join(output_dir, scene_name), exist_ok=True)
        for idx, (overlap, det) in enumerate(zip(overlaps, original_dets)):
            save_file_name = os.path.join(output_dir, scene_name,
                                          'detection_3d_{:03d}.txt'.format(idx))
            save_file = open(save_file_name, 'w')
            str = 'class,x,y,z,w(l_x),l(l_y),h(l_z),yaw,iou,score,num_points \n'
            save_file.write(str)
            ious = np.amax(overlap, axis=0)
            for iou, obj in zip(ious, det):
                str = format_sample_result(obj, iou)
                save_file.write(str)
            save_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str)
    parser.add_argument('--det_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    main(args.gt_dir, args.det_dir, args.output_dir)