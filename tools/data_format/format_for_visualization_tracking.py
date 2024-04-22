import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from nuscenes import NuScenes
from copy import deepcopy
import torch
import pickle
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

# import debugpy
# debugpy.listen(7777)
# print("Wait for Debugger for format_for_vis_tracking.py")
# debugpy.wait_for_client()
# print("Debugger Attached")

# v1.0-mini
# nusc = NuScenes(version='v1.0-mini', dataroot='/data_from_host/shared/datasets/nuscenes/nuscenes_mini_with_prev_next/', verbose=True)

# v1.0-trainval
nusc = NuScenes(version='v1.0-trainval', dataroot='/data_from_host/shared/datasets/nuscenes/nuscenes_full_with_prev_next_container_bevfusion_new/', verbose=True)

def transform_to_lidar(translation, rotation, velocity, lidar2car, car2world):
    
    """
    convert translation, rotation and velocity to lidar coordinates in 
    order to visualize. Desired format is the same as detection results of 
    BEVFusion

    args: 
            translation: nparray (1, 3)
            rotation: nparray (1, 4)
            velocity: nparray (1, 2)

    return:
            old_trans: nparray (1, 3)
            old_rot: nparray (1, 4)
            old_vel: nparray (1, 2)

    """
    
    trans_in_mat = np.array(translation).reshape((3,1))   # 3*1

    xyzw = wxyz_to_xyzw(rotation)
    rot_in_mat = R.from_quat(xyzw).as_matrix() # 3*3

    trans = np.hstack([rot_in_mat, trans_in_mat])
    trans = np.vstack([trans, np.array([0, 0, 0, 1])]) # 4*4

    velo = np.hstack([velocity, np.array([0])]).reshape((3, 1)) # 3*1

    lidar2car_inv = np.linalg.inv(lidar2car)
    car2world_inv = np.linalg.inv(car2world)

    # ======================================================== #

    old_mat = lidar2car_inv.dot(car2world_inv.dot(trans)) # 4*4

    old_trans = old_mat[:3,3] # 1*3
    old_rot = R.from_matrix(old_mat[:3,:3]).as_quat() # 1*4 # rotation_matrix to quaternion
    old_rot = q_to_wxyz(old_rot)

    old_vel = lidar2car_inv[:3, :3].dot(car2world_inv[:3, :3].dot(velo))[:2].ravel()
    
    return old_trans, old_rot, old_vel

def wxyz_to_xyzw(Q):
    '''
    xyzw -> wxyz
    '''
    return [Q[1], Q[2], Q[3], Q[0]]

def q_to_wxyz(Q):
    '''
    xyzw -> wxyz
    '''
    return [Q[3], Q[0], Q[1], Q[2]]


def get_4f_transform(pose, inverse=False):
    return transform_matrix(pose['translation'], Quaternion(pose['rotation']), inverse=inverse)


def main():
    
    # step 0: load tracking result
    f = open('results/tracking_centerpoint_first_with_residual_decoder_epoch_bbox_thres_0.01/tracking_result.json') # 'results/tracking_full_2nd_3_epoch/tracking_result.json'
    res = json.load(f)

    tracking_res = deepcopy(res)
    for sample_token, bbox in tracking_res['results'].items():
        sample_record = nusc.get('sample', sample_token)
        LIDAR_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', LIDAR_record['ego_pose_token'])
        cs_record = nusc.get('calibrated_sensor', LIDAR_record['calibrated_sensor_token'])
        lidar2car = get_4f_transform(cs_record, inverse=False)
        car2world = get_4f_transform(ego_pose, inverse=False)
        
        for bbox_idx in range(len(bbox)):
            translation = np.array(bbox[bbox_idx]['translation'])
            rotation = bbox[bbox_idx]['rotation']
            velocity = bbox[bbox_idx]['velocity']
            new_trans, new_rot, new_vel = transform_to_lidar(translation, rotation, velocity, lidar2car, car2world)
            yaw = R.from_quat(new_rot).as_euler('zxy', degrees=True)[1]
            yaw = - (yaw + np.pi / 2)  
            bbox[bbox_idx]['translation'] = new_trans
            bbox[bbox_idx]['rotation'] = yaw
            bbox[bbox_idx]['velocity'] = new_vel
            bbox[bbox_idx]['size'] = np.array(bbox[bbox_idx]['size'])
            
    tracking_dict = {}
    for sample_token, bbox in tracking_res['results'].items():
        if len(bbox) == 0: # no box in sample
            continue
        sample_idx = bbox[0]['sample_token']
        boxes_3d = []
        labels_3d = []
        scores_3d = []
        tracking_name = []
        temp_dict = {}
        for bbox_idx in range(len(bbox)):
            new_tensor = np.hstack([bbox[bbox_idx]['translation'], bbox[bbox_idx]['size'], bbox[bbox_idx]['rotation'], bbox[bbox_idx]['velocity']])
            bbox[bbox_idx]['tensor'] = torch.tensor(new_tensor)
            boxes_3d.append(torch.tensor(new_tensor))
            # print('tracking id is: ', bbox[bbox_idx]['tracking_id'])
            labels_3d.append(torch.tensor(int(bbox[bbox_idx]['tracking_id'])))
            # print('converted into: ', labels_3d[bbox_idx])
            scores_3d.append(torch.tensor(bbox[bbox_idx]['tracking_score'])) 
            tracking_name.append(bbox[bbox_idx]['tracking_name'])
            # print('tracking_name is: ', tracking_name)
            # break
        # break
        temp_dict['boxes_3d'] = torch.stack(boxes_3d)
        temp_dict['scores_3d'] = torch.stack(scores_3d)
        temp_dict['labels_3d'] = torch.stack(labels_3d)
        temp_dict['tracking_name'] = tracking_name
        tracking_dict[sample_idx] = temp_dict
    
    # save in .pkl    
    with open('results/tracking_centerpoint_first_with_residual_decoder_epoch_bbox_thres_0.01/tracking_result.pkl', 'wb') as handle: # 'results/visualization_tracking_full_2nd_3_epoch/visualization_tracking_full_2nd_3_epoch.pkl'
        pickle.dump(tracking_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
if __name__ == '__main__':
    main()