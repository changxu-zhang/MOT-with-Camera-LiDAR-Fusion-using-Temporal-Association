# MOT with Camera-LiDAR Fusion using Temporal Association

## Abstract

## TODO
- [x] complete readme
- [x] detector and tracker
- [x] script for format conversion
- [] upload checkpoint in drive
- [] provide docker file/image

## Usage
### Prerequisites
Following libraries are necessary for running our codes:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0
- PyTorch >= 1.9, \<= 1.10.2
- pandas>=0.24
- numba
- numpy
- sympy
- tqdm
- torchpack
- mmcv == 1.4.0
- mmdetection == 2.20.0
- nuscenes-devkit
- motmetrics==1.1.3
- pyquaternion
- line_profiler

### Train & Evaluation
Train the detector with pre-trained model:
```bash
torchpack dist-run -np [number of gpus] python tools/train.py [path_to_config] --load_from [path_to_pretrained]
```

Evaluate the detector with specific type (e.g. bbox) and save results:
```bash
torchpack dist-run -np [number of gpus] python tools/test.py [path_to_config] [path_to_checkpoint.pth] --eval [evaluation type] --out [path_to_result.pkl]
```

Convert the format of detection result to adapt with tracker:
```bash
cd script/
python format_for_tracker.py
```
Note that path of detection result and destination need to be specified in the file.

Run the CenterPoint tracker:
```bash
python track/centerpoint/track_test.py --checkpoint [path_to_detection_result.json]  --work_dir [path_to_result] --bbox-score [confidence e.g. 0.01]
```

Run the Poly-MOT tracker:
```bash
python test.py --detection_path [path_to_detection_result.json] --eval_path [path_to_result] 
```


### Visualization
Visualize detection result:
```bash
torchpack dist-run -np [number of gpus] python tools/visualize.py [path_to_config] --checkpoint [path_to_checkpoint.pth] --out-dir [path_to_vis_result] --mode [pred or gt] --bbox-score [confidence e.g. 0.01]
```

We need to convert data format using (Note that path of detection result and destination need to be specified in the file):
```bash
cd script/
python format_for_tracking_visual.py
```

Then, we can visualize tracking results:
```bash
python tools/track/visualize_tracking.py [path_to_config] --checkpoint [path_to_tracking_result.pkl] --out-dir [path_to_vis_result] --mode [pred or gt] --bbox-score [confidence e.g. 0.01]
```

## Result
[Checkpoints](https://tubcloud.tu-berlin.de/s/kNrqQifZmNia3Hr) are available.

[Detection](https://tubcloud.tu-berlin.de/s/W5TfWb5pFZZnXpq) and [Tracking](https://tubcloud.tu-berlin.de/s/9zxGQKnQy6sHrTn) results are also provided.
