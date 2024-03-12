import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import json

# v1.0-mini
# nusc = NuScenes(version='v1.0-mini', dataroot='/data_from_host/shared/datasets/nuscenes/nuscenes_mini_with_prev_next/', verbose=True)

# v1.0-trainval
nusc = NuScenes(version='v1.0-trainval', dataroot='/data_from_host/shared/datasets/nuscenes/nuscenes_full_with_prev_next_container_bevfusion_new/', verbose=True)

def yaw_to_quaternion(yaw):
    yaw = - yaw - (np.pi / 2) # TODO: why
    rot = Rotation.from_euler('xyz', [0, 0, yaw], degrees=True)
    rot_quat = rot.as_quat()
    return [rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]]

def format_centerpoint(res_bevfusion):
    new_res = {}
    for sample_idx in range(len(res_bevfusion)):
        sample_token = res_bevfusion[sample_idx]['sample_token']
        new_bboxes = []
        for bbox_idx in range(res_bevfusion[sample_idx]['boxes_3d'].tensor.shape[0]):
            new_bbox = {}
            translation = res_bevfusion[sample_idx]['boxes_3d'].tensor[bbox_idx, :3]
            size = res_bevfusion[sample_idx]['boxes_3d'].tensor[bbox_idx, 3:6]
            rotation = res_bevfusion[sample_idx]['boxes_3d'].tensor[bbox_idx, 6]
            velocity = res_bevfusion[sample_idx]['boxes_3d'].tensor[bbox_idx, 7:]
            detection_name = res_bevfusion[sample_idx]['labels_3d'][bbox_idx]
            detection_score = res_bevfusion[sample_idx]['scores_3d'][bbox_idx]
            attribute_name = ''
            new_bbox['sample_token'] = sample_token
            new_bbox['translation'] = translation.tolist()
            new_bbox['size'] = size.tolist()
            new_bbox['rotation'] = yaw_to_quaternion(rotation.tolist())
            new_bbox['velocity'] = velocity.tolist()
            new_bbox['detection_name'] = detection_name.tolist()
            new_bbox['detection_score'] = detection_score.tolist()
            new_bbox['attribute_name'] = attribute_name
            new_bboxes.append(new_bbox)
        new_res[sample_token] = new_bboxes         
    return new_res 

def get_4f_transform(pose, inverse=False):
    return transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=inverse)

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)  (w,x,y,z)
    
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
        
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
        
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
        
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def q_to_wxyz(Q):
    '''
    xyzw -> wxyz
    '''
    return [Q[3], Q[0], Q[1], Q[2]]

def lidar2world(objects, token, inverse=False):
    '''
    Transform objects from lidar coordinates to world coordinates
    '''
    objects = deepcopy(objects)
    
    sample_record = nusc.get('sample', token)
    LIDAR_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', LIDAR_record['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', LIDAR_record['calibrated_sensor_token'])
    lidar2car = get_4f_transform(cs_record, inverse=inverse)
    car2world = get_4f_transform(ego_pose, inverse=inverse)

    ret = []
    for object in objects:
        trans = np.array(object['translation'])
        vel = np.array([object['velocity'][0], object['velocity'][1], 0.0])
        rot = quaternion_rotation_matrix(object['rotation'])
        trans = np.hstack([rot, trans.reshape(-1, 1)])
        trans = np.vstack([trans, np.array([0, 0, 0, 1])]).reshape(-1, 4)
        vel = vel.reshape(-1, 1)
        if not inverse:
            new_trans = car2world.dot(lidar2car.dot(trans))
            new_vel = car2world[:3, :3].dot(lidar2car[:3, :3].dot(vel))
        elif inverse:
            new_trans = lidar2car.dot(car2world.dot(trans))
            new_vel = lidar2car[:3, :3].dot(car2world[:3, :3].dot(vel))
        object['translation'] = new_trans[:3, 3].ravel().tolist()
        object['rotation'] = q_to_wxyz(R.from_matrix(new_trans[:3, :3]).as_quat())
        object['velocity'] = new_vel.ravel()[:2].tolist()
        ret.append(object)

    return ret

def format_centerpoint_new(temp_res):
    centerpoint_res = {}
    for sample_token, bboxes in temp_res.items():
        new_bboxes = lidar2world(bboxes, sample_token)      
        centerpoint_res[sample_token] = new_bboxes         
    return centerpoint_res 


def main():

    # step 0: load BEVFusion detection results
    res_bevfusion = pd.read_pickle('results/2nd_with_full_dataset_3_epoch.pkl')

    # step 1: format the nest structure
    temp_res = format_centerpoint(res_bevfusion) # get temporary result for next step

    # step 2: coordinate transformation for translation, rotation and velocity
    centerpoint_res = format_centerpoint_new(temp_res) # detection results in CenterPoint format

    # step 3: wrap results with metas
    new_ret = {}
    metas = {
        'use_camera': True,
        'use_lidar': True,
        'use_radar': False,
        'use_map': False,
        'use_external': False
    }
    new_ret['results'] = centerpoint_res 
    
    # Tracking with CenterPoint, using "metas" 
    # new_ret['metas'] = metas
    # Tracking with Poly-MOT, using "meta" 
    new_ret['meta'] = metas

    # step 4: save new results in .json
    save_file = open('results/2nd_with_full_dataset_3_epoch_polymot.json', 'w')
    json.dump(new_ret, save_file)
    save_file.close()

if __name__ == '__main__':
    main()
