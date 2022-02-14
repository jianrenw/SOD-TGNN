import numpy as np
import os
import os.path as osp
from numpy.linalg import inv


def rotx(t):
    ''' Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def get_matrix(odom):
    rotation = np.matmul(np.matmul(rotz(odom[5]), roty(odom[4])),
                         rotx(odom[3]))
    translation = np.transpose(np.array([odom[0], odom[1], odom[2]]))
    tf_matrix = np.identity(4)
    tf_matrix[:3, :3] = rotation
    tf_matrix[:3, 3] = translation
    # print(tf_matrix)
    return tf_matrix

def get_gps(gps_file):
    lines = open(gps_file,'r').readlines()
    x = 0.
    y = 0.
    z = 0.
    roll = 0.
    pitch = 0.
    yaw = 0.
    size = len(lines)-1
    for i in range(1, len(lines)):
        component = lines[i].split(',')
        if 'gps' in gps_file:
            x += float(component[1])
            y += float(component[2])
        else:
            x += float(component[1])
            y += float(component[2])
        z += float(component[3])
        roll += float(component[4])*np.pi/180.
        pitch += float(component[5])*np.pi/180.
        yaw += float(component[6])*np.pi/180.
    # f_x, f_y, time_zone = LLtoUTM(x/size, y/size)
    # print(f_x, f_y, time_zone)
    # gps = [f_x, f_y, z/size, (roll/size), pitch/size, yaw/size]
    gps = [x/size, -1.0*y/size, 0., roll/size, -1.0*pitch/size, yaw/size]
    # gps = [x/size, y/size, -1.0*z/size, roll/size, pitch/size, yaw/size]
    # print(gps)
    return gps

class h3d(object):
    def __init__(self, gt_dir, det_dir, projection):
        self.gt_dir = gt_dir
        self.det_dir = det_dir
        self.projection = projection
        self.class_index = {
            'Car': 0,
            'Pedestrian': 1,
            'Cyclist': 2,
            'Other vehicle': 3,
            'Bus': 4,
            'Truck': 5,
            'Motorcyclist': 6,
            'Animals': 7
        }
        self.scene_names = sorted(os.listdir(self.det_dir))
    
    def get_gts_odom(self, scene_name):
        gts = []
        file_dir = osp.join(self.gt_dir, scene_name)
        file_names = os.listdir(file_dir)
        label_names = sorted(
            [file_name for file_name in file_names if 'labels' in file_name])
        odom_names = sorted(
            [file_name for file_name in file_names if 'odom' in file_name])
        odom_info = np.loadtxt(osp.join(file_dir, odom_names[0]),
                               delimiter=' ',
                               dtype=float)
        first_matrix = get_matrix(odom_info)
        initial_matrix = inv(first_matrix)
        for label_name, odom_name in zip(label_names, odom_names):
            label_file = osp.join(file_dir, label_name)
            label = np.loadtxt(label_file, delimiter=',', dtype='str')
            odom_file = osp.join(file_dir, odom_name)
            odom_info = np.loadtxt(odom_file, delimiter=' ', dtype=float)
            origin_matrix = get_matrix(odom_info)
            new_matrix = np.matmul(initial_matrix, origin_matrix)
            positions = label[:, 3:6].astype(np.float)
            l = label[:, 6:7].astype(np.float)
            w = label[:, 7:8].astype(np.float)
            h = label[:, 8:9].astype(np.float)
            yaw = label[:, 9:].astype(np.float)
            category = np.zeros((positions.shape[0],1))
            for i in range(positions.shape[0]):
                orientation = np.array([0., 0., yaw[i, 0]])
                pos_info = np.concatenate((positions[i, :], orientation))
                curr_matrix = get_matrix(pos_info)
                new_position = np.matmul(new_matrix, curr_matrix)
                new_pos = new_position[:3, 3]
                new_yaw = np.arctan2(new_position[1, 0], new_position[0, 0])
                positions[i, :] = new_pos
                yaw[i, 0] = new_yaw
                category[i, 0] = self.class_index[label[i,0]]
            gts.append(np.concatenate((category, h, w, l, positions, yaw), axis=1))
        return gts

    def get_dets_odom(self, scene_name):
        dets = []
        det_file_dir = osp.join(self.det_dir, scene_name)
        file_names = os.listdir(det_file_dir)
        det_names = sorted([
            file_name for file_name in file_names if 'detection' in file_name
        ])
        odom_file_dir = osp.join(self.gt_dir, scene_name)
        file_names = os.listdir(odom_file_dir)
        odom_names = sorted(
            [file_name for file_name in file_names if 'odom' in file_name])
        odom_info = np.loadtxt(osp.join(odom_file_dir, odom_names[0]),
                               delimiter=' ',
                               dtype=float)
        first_matrix = get_matrix(odom_info)
        initial_matrix = inv(first_matrix)
        for det_name, odom_name in zip(det_names, odom_names):
            det_file = osp.join(det_file_dir, det_name)
            det = np.loadtxt(det_file, delimiter=',', dtype='str')[1:, :]
            odom_file = osp.join(odom_file_dir, odom_name)
            odom_info = np.loadtxt(odom_file, delimiter=' ', dtype=float)
            origin_matrix = get_matrix(odom_info)
            new_matrix = np.matmul(initial_matrix, origin_matrix)
            positions = det[:, 1:4].astype(np.float)
            l = det[:, 4:5].astype(np.float)
            w = det[:, 5:6].astype(np.float)
            h = det[:, 6:7].astype(np.float)
            yaw = det[:, 7:8].astype(np.float)
            confidence = det[:, 8:9].astype(np.float)
            category = np.zeros((positions.shape[0],1))
            for i in range(positions.shape[0]):
                orientation = np.array([0., 0., yaw[i, 0]])
                pos_info = np.concatenate((positions[i, :], orientation))
                curr_matrix = get_matrix(pos_info)
                new_position = np.matmul(new_matrix, curr_matrix)
                new_pos = new_position[:3, 3]
                new_yaw = np.arctan2(new_position[1, 0], new_position[0, 0])
                positions[i, :] = new_pos
                yaw[i, 0] = new_yaw
                category[i, 0] = self.class_index[det[i,0]]
            dets.append(np.concatenate((category, confidence, h, w, l, positions, yaw), axis=1))
        return dets

    def get_gts_gps(self, scene_name):
        gts = []
        file_dir = osp.join(self.gt_dir, scene_name)
        file_names = os.listdir(file_dir)
        label_names = sorted(
            [file_name for file_name in file_names if 'labels' in file_name])
        gps_names = sorted(
            [file_name for file_name in file_names if 'gps' in file_name])
        gps_info = get_gps(osp.join(file_dir, gps_names[0]))
        first_matrix = get_matrix(gps_info)
        initial_matrix = inv(first_matrix)
        for label_name, gps_name in zip(label_names, gps_names):
            label_file = osp.join(file_dir, label_name)
            label = np.loadtxt(label_file, delimiter=',', dtype='str')
            gps_file = osp.join(file_dir, gps_name)
            gps_info = get_gps(gps_file)
            origin_matrix = get_matrix(gps_info)
            new_matrix = np.matmul(initial_matrix, origin_matrix)
            positions = label[:, 3:6].astype(np.float)
            l = label[:, 6:7].astype(np.float)
            w = label[:, 7:8].astype(np.float)
            h = label[:, 8:9].astype(np.float)
            yaw = label[:, 9:].astype(np.float)
            category = np.zeros((positions.shape[0],1))
            for i in range(positions.shape[0]):
                orientation = np.array([0., 0., yaw[i, 0]])
                pos_info = np.concatenate((positions[i, :], orientation))
                curr_matrix = get_matrix(pos_info)
                new_position = np.matmul(new_matrix, curr_matrix)
                new_pos = new_position[:3, 3]
                new_yaw = np.arctan2(new_position[1, 0], new_position[0, 0])
                positions[i, :] = new_pos
                yaw[i, 0] = new_yaw
                category[i, 0] = self.class_index[label[i,0]]
            gts.append(np.concatenate((category, h, w, l, positions, yaw), axis=1))
        return gts

    def get_dets_gps(self, scene_name):
        dets = []
        det_file_dir = osp.join(self.det_dir, scene_name)
        file_names = os.listdir(det_file_dir)
        det_names = sorted([
            file_name for file_name in file_names if 'detection' in file_name
        ])
        gps_file_dir = osp.join(self.gt_dir, scene_name)
        file_names = os.listdir(gps_file_dir)
        gps_names = sorted(
            [file_name for file_name in file_names if 'gps' in file_name])
        gps_info = get_gps(osp.join(gps_file_dir, gps_names[0]))
        first_matrix = get_matrix(gps_info)
        initial_matrix = inv(first_matrix)
        for det_name, gps_name in zip(det_names, gps_names):
            det_file = osp.join(det_file_dir, det_name)
            det = np.loadtxt(det_file, delimiter=',', dtype='str')[1:, :]
            gps_file = osp.join(gps_file_dir, gps_name)
            gps_info = get_gps(gps_file)
            origin_matrix = get_matrix(gps_info)
            new_matrix = np.matmul(initial_matrix, origin_matrix)
            positions = det[:, 1:4].astype(np.float)
            l = det[:, 4:5].astype(np.float)
            w = det[:, 5:6].astype(np.float)
            h = det[:, 6:7].astype(np.float)
            yaw = det[:, 7:8].astype(np.float)
            confidence = det[:, 8:9].astype(np.float)
            category = np.zeros((positions.shape[0],1))
            for i in range(positions.shape[0]):
                orientation = np.array([0., 0., yaw[i, 0]])
                pos_info = np.concatenate((positions[i, :], orientation))
                curr_matrix = get_matrix(pos_info)
                new_position = np.matmul(new_matrix, curr_matrix)
                new_pos = new_position[:3, 3]
                new_yaw = np.arctan2(new_position[1, 0], new_position[0, 0])
                positions[i, :] = new_pos
                yaw[i, 0] = new_yaw
                category[i, 0] = self.class_index[det[i,0]]
            dets.append(np.concatenate((category, confidence, h, w, l, positions, yaw), axis=1))
        return dets

    def get_gts_original(self, scene_name):
        gts = []
        file_dir = osp.join(self.gt_dir, scene_name)
        file_names = os.listdir(file_dir)
        label_names = sorted(
            [file_name for file_name in file_names if 'labels' in file_name])
        for label_name in label_names:
            label_file = osp.join(file_dir, label_name)
            label = np.loadtxt(label_file, delimiter=',', dtype='str')
            positions = label[:, 3:6].astype(np.float)
            l = label[:, 6:7].astype(np.float)
            w = label[:, 7:8].astype(np.float)
            h = label[:, 8:9].astype(np.float)
            yaw = label[:, 9:].astype(np.float)
            category = np.zeros((positions.shape[0],1))
            for i in range(positions.shape[0]):
                category[i, 0] = self.class_index[label[i,0]]
            gts.append(np.concatenate((category, h, w, l, positions, yaw), axis=1))
        return gts

    def get_dets_original(self, scene_name):
        dets = []
        det_file_dir = osp.join(self.det_dir, scene_name)
        file_names = os.listdir(det_file_dir)
        det_names = sorted([
            file_name for file_name in file_names if 'detection' in file_name
        ])
        for det_name in det_names:
            det_file = osp.join(det_file_dir, det_name)
            det = np.loadtxt(det_file, delimiter=',', dtype='str')[1:, :]
            positions = det[:, 1:4].astype(np.float)
            l = det[:, 4:5].astype(np.float)
            w = det[:, 5:6].astype(np.float)
            h = det[:, 6:7].astype(np.float)
            yaw = det[:, 7:8].astype(np.float)
            confidence = det[:, 8:9].astype(np.float)
            category = np.zeros((positions.shape[0],1))
            for i in range(positions.shape[0]):
                category[i, 0] = self.class_index[det[i,0]]
            dets.append(np.concatenate((category, confidence, h, w, l, positions, yaw), axis=1))
        return dets

    def get_dets(self, scene_name):
        if self.projection == 'odom':
            return self.get_dets_odom(scene_name)
        elif self.projection == 'gps':
            return self.get_dets_gps(scene_name)
        elif self.projection == 'original':
            return self.get_dets_original(scene_name)

    def get_gts(self, scene_name):
        if self.projection == 'odom':
            return self.get_gts_odom(scene_name)
        elif self.projection == 'gps':
            return self.get_gts_gps(scene_name)
        elif self.projection == 'original':
            return self.get_gts_original(scene_name)

    def get_original_dets(self, scene_name):
        dets = []
        file_dir = osp.join(self.det_dir, scene_name)
        file_names = os.listdir(file_dir)
        det_names = sorted([
            file_name for file_name in file_names if 'detection' in file_name
        ])
        for det_name in det_names:
            det_file = osp.join(file_dir, det_name)
            det = np.loadtxt(det_file, delimiter=',', dtype='str')
            dets.append(det)
        return dets