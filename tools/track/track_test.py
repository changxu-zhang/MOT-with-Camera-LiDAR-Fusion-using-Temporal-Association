# copy-paste from CenterPoint https://github.com/tianweiy/CenterPoint/blob/master/tools/nusc_tracking/pub_test.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys 
import json
import numpy as np
import time
import argparse
import copy
import json
import os
import numpy as np
from pub_tracker import PubTracker as Tracker
from nuscenes import NuScenes
import json 
import time
from nuscenes.utils import splits
from tqdm import tqdm

# import debugpy
# debugpy.listen(5555)
# print("Wait for Debugger for track_test.py")
# debugpy.wait_for_client()
# print("Debugger Attached")

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking Evaluation")
    parser.add_argument("--work_dir", help="the dir to save logs and tracking results")
    parser.add_argument("--checkpoint", help="the dir to checkpoint which the model read from")
    parser.add_argument("--hungarian", action='store_true')
    parser.add_argument("--root", type=str, default="/data_from_host/shared/datasets/nuscenes/nuscenes_full_with_prev_next_container_bevfusion_new") # nuscenes_mini_with_prev_next # nuscenes_full_with_prev_next_container_bevfusion_new
    parser.add_argument("--version", type=str, default='v1.0-trainval') # v1.0-trainval # v1.0-mini # v1.0-test
    parser.add_argument("--max_age", type=int, default=3)
    parser.add_argument("--bbox_score", type=float, default=None, help="to filter out bbox with low confidence from detection")

    args = parser.parse_args()

    return args


def save_first_frame():
    args = parse_args()
    nusc = NuScenes(version=args.version, dataroot=args.root, verbose=True)
    if args.version == 'v1.0-trainval':
        scenes = splits.val
    elif args.version == 'v1.0-test':
        scenes = splits.test 
    elif args.version == 'v1.0-mini':
        scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    frames = []
    for sample in nusc.sample:
        scene_name = nusc.get("scene", sample['scene_token'])['name'] 
        if scene_name not in scenes:
            continue 

        timestamp = sample["timestamp"] * 1e-6
        token = sample["token"]
        frame = {}
        frame['token'] = token
        frame['timestamp'] = timestamp 

        # start of a sequence
        if sample['prev'] == '':
            frame['first'] = True 
        else:
            frame['first'] = False 
        frames.append(frame)

    del nusc

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    with open(os.path.join(args.work_dir, 'frames_meta.json'), "w") as f:
        json.dump({'frames': frames}, f)


def main():
    args = parse_args()
    print('Deploy OK')

    tracker = Tracker(max_age=args.max_age, hungarian=args.hungarian)

    with open(args.checkpoint, 'rb') as f:
        predictions=json.load(f)['results']

    with open(os.path.join(args.work_dir, 'frames_meta.json'), 'rb') as f:
        frames=json.load(f)['frames']

    nusc_annos = {
        "results": {},
        "meta": None,
    }
    size = len(frames)

    print("Begin Tracking\n")
    start = time.time()
    for i in tqdm(range(size)):
        token = frames[i]['token']

        # reset tracking after one video sequence
        if frames[i]['first']:
            # use this for sanity check to ensure your token order is correct
            # print("reset ", i)
            tracker.reset()
            last_time_stamp = frames[i]['timestamp']

        time_lag = (frames[i]['timestamp'] - last_time_stamp) 
        last_time_stamp = frames[i]['timestamp']

        preds = predictions[token] # pred must be a dict with sample_token as key

        # filter out detections with low confidence
        if args.bbox_score is not None:
            preds_filtered = [bbox for bbox in preds if bbox['detection_score'] >= args.bbox_score]
        else:
            preds_filtered = preds
        
        outputs = tracker.step_centertrack(preds_filtered, time_lag)
        annos = []

        for item in outputs:
            if item['active'] == 0:
                continue 
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "tracking_id": str(item['tracking_id']),
                "tracking_name": item['detection_name'],
                "tracking_score": item['detection_score'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})

    
    end = time.time()

    second = (end-start) 

    speed=size / second
    print("The speed is {} FPS".format(speed))

    nusc_annos["meta"] = {
        "use_camera": True,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    res_dir = os.path.join(args.work_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    with open(os.path.join(args.work_dir, 'tracking_result.json'), "w") as f:
        json.dump(nusc_annos, f)
    return speed

def eval_tracking():
    args = parse_args()
    eval_set = ""
    if args.version == 'v1.0-trainval':
        eval_set = 'val'
    else: # args.version == 'v1.0-mini'
        eval_set = 'mini_val'
        
    eval(os.path.join(args.work_dir, 'tracking_result.json'),
        eval_set, # "val" # mini_val
        args.work_dir,
        args.root
    )

def eval(res_path, eval_set="val", output_dir=None, root_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval 
    from nuscenes.eval.common.config import config_factory as track_configs

    args = parse_args()
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version=args.version, # v1.0-trainval # v1.0-mini
        nusc_dataroot=root_path,
    )
    metrics_summary = nusc_eval.main()


def test_time():
    speeds = []
    for i in range(3):
        speeds.append(main())

    print("Speed is {} FPS".format( max(speeds)  ))

if __name__ == '__main__':
    save_first_frame()
    main()
    # test_time()
    eval_tracking()