from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet.models.backbones.swin import SwinTransformer

from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if encoders.get("radar") is not None:
            if encoders["radar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["radar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["radar"]["voxelize"])
            self.encoders["radar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["radar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["radar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        
        '''
        start of modification
        '''
        # temporal decoder
        # self.red_conv = nn.Conv2d(512, 256, (1, 1), 1)
        self.red_pool = nn.MaxPool2d(3, 3, return_indices=True)
        self.unpool = nn.MaxUnpool2d(3, 3, padding=0)
        
        self.red_pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2, padding=0)
        self.red_pool_key = nn.MaxPool2d(3, 3, return_indices=True)
        
        self.temporal_decoder = TransformerDecoderLayer(
            d_model=512,
            nhead=8,    
        )
       
        # copy-paste from def create_2D_grid in transfusion to build pos embedding for decoder
        x_size = 90 # 60
        y_size = 90 # 60
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        self.x_pos = coord_base.view(1, 2, -1).permute(0, 2, 1)
        
        x_size = 60
        y_size = 60
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        self.x_pos_key = coord_base.view(1, 2, -1).permute(0, 2, 1)
        
        # init x_query
        self.x_query = torch.zeros((4, 512 ,200), device='cuda')
        self.query_pos = torch.zeros((4, 200, 2), device='cuda')
        '''
        end of modification
        '''
        
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # If the camera's vtransform is a BEVDepth version, then we're using depth loss. 
        self.use_depth_loss = ((encoders.get('camera', {}) or {}).get('vtransform', {}) or {}).get('type', '') in ['BEVDepth', 'AwareBEVDepth', 'DBEVDepth', 'AwareDBEVDepth']


        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        gt_depths=None,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            radar_points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
            depth_loss=self.use_depth_loss, 
            gt_depths=gt_depths,
        )
        return x
    
    def extract_features(self, x, sensor) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x
    
    # def extract_lidar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    # def extract_radar_features(self, x) -> torch.Tensor:
    #     feats, coords, sizes = self.radar_voxelize(x)
    #     batch_size = coords[-1, 0] + 1
    #     x = self.encoders["radar"]["backbone"](feats, coords, batch_size, sizes=sizes)
    #     return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    # @torch.no_grad()
    # @force_fp32()
    # def radar_voxelize(self, points):
    #     feats, coords, sizes = [], [], []
    #     for k, res in enumerate(points):
    #         ret = self.encoders["radar"]["voxelize"](res)
    #         if len(ret) == 3:
    #             # hard voxelize
    #             f, c, n = ret
    #         else:
    #             assert len(ret) == 2
    #             f, c = ret
    #             n = None
    #         feats.append(f)
    #         coords.append(F.pad(c, (1, 0), mode="constant", value=k))
    #         if n is not None:
    #             sizes.append(n)

    #     feats = torch.cat(feats, dim=0)
    #     coords = torch.cat(coords, dim=0)
    #     if len(sizes) > 0:
    #         sizes = torch.cat(sizes, dim=0)
    #         if self.voxelize_reduce:
    #             feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
    #                 -1, 1
    #             )
    #             feats = feats.contiguous()

    #     return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                depths,
                radar,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        bs, len_queue, num_cams, C, H, W = img.shape
        img = img.reshape(bs*len_queue, num_cams, C, H, W)
        # self.save_tensor(img, metas[0][1]['token']+'_img.pt')
        
        _, _, num_pc, pc = points.shape
        points = points.reshape(bs*len_queue, num_pc, pc)
        # self.save_tensor(points, metas[0][1]['token']+'_points.pt')
        
        # flatten nested list
        for bs_idx in range(bs):
            # for queue_idx in range(len_queue):
            gt_bboxes_3d.append(gt_bboxes_3d[bs_idx][-1])
            gt_labels_3d.append(gt_labels_3d[bs_idx][-1].to(img.device))
            metas.append(metas[bs_idx][len_queue - 1]) # only works when len_queue=2
        gt_bboxes_3d = gt_bboxes_3d[bs:]
        gt_labels_3d = gt_labels_3d[bs:]
        # TODO: better workaround
        # gt_bboxes_3d_removed_1st = gt_bboxes_3d.pop(0)
        # gt_bboxes_3d_removed_2nd = gt_bboxes_3d.pop(1)
        # gt_labels_3d_removed_1st = gt_labels_3d.pop(0)
        # gt_labels_3d_removed_2nd = gt_labels_3d.pop(1)
        metas = metas[bs:]
        
        # comment out lines with stack, if DC(..., cpu_only=False) in mmdet3d/datasets/custom_3d.py/union2one
        # list of bs to (bs, queue_length, ...) then (bs*queue_length, ...)
        # camera2ego = torch.stack(camera2ego, dim=0) 
        camera2ego = camera2ego.reshape(bs*len_queue, num_cams, 4, 4)
        
        # lidar2ego = torch.stack(lidar2ego, dim=0) 
        lidar2ego = lidar2ego.reshape(bs*len_queue, 4, 4)
        
        # lidar2camera = torch.stack(lidar2camera, dim=0)
        lidar2camera = lidar2camera.reshape(bs*len_queue, num_cams, 4, 4)
         
        # lidar2image = torch.stack(lidar2image, dim=0) 
        lidar2image = lidar2image.reshape(bs*len_queue, num_cams, 4, 4)
        
        # camera_intrinsics = torch.stack(camera_intrinsics, dim=0) 
        camera_intrinsics = camera_intrinsics.reshape(bs*len_queue, num_cams, 4, 4)
        
        # camera2lidar = torch.stack(camera2lidar, dim=0)
        camera2lidar = camera2lidar.reshape(bs*len_queue, num_cams, 4, 4)
        
        # img_aug_matrix = torch.stack(img_aug_matrix, dim=0)
        img_aug_matrix = img_aug_matrix.reshape(bs*len_queue, num_cams, 4, 4)
        
        # lidar_aug_matrix = torch.stack(lidar_aug_matrix, dim=0)
        lidar_aug_matrix = lidar_aug_matrix.reshape(bs*len_queue, 4, 4)
        
        features = []
        auxiliary_losses = {}
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    radar,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    gt_depths=depths,
                )
                if self.use_depth_loss:
                    feature, auxiliary_losses['depth'] = feature[0], feature[-1]
            elif sensor == "lidar":
                feature = self.extract_features(points, sensor)
            elif sensor == "radar":
                feature = self.extract_features(radar, sensor)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")

            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x) # [4, 512, 180, 180]
        
        # split current and history features TODO: better workaround for different bs
        x = x[0]
        if bs == 2:
            x_current = torch.stack((x[len_queue - 1, ...], x[len_queue * bs - 1, ...]), dim=0) # only works for bs=2 right now
            x_history = torch.cat((x[:len_queue - 1, ...], x[len_queue:len_queue * bs - 1, ...]), dim=0)
        else: # bs == 1:
            x_current = x[len_queue - 1, ...].unsqueeze(0)
            x_history = x[:len_queue - 1, ...]

        # reduce size of bev h, w to ovoid oom
        x_current, indices_pool_current = self.red_pool(x_current) 
        x_history, indices_pool_history = self.red_pool_key(x_history) # stride=3 (4, 512, 60, 60)
        c, h, w = x_current.shape[-3:]
        
        # flatten before input into temporal
        x_flatten = x[0].view(batch_size, c, -1) # [BS, C, H*W]
        x_current_flatten = x_current.view(batch_size//len_queue, x_current.shape[1], -1)
        x_history_flatten = x_history.view(batch_size//len_queue, x_history.shape[1], -1)

        x_pos = self.x_pos.repeat(batch_size // 2, 1, 1).to(x_flatten.device)
        x_pos = self.x_pos.repeat(batch_size // 2, 1, 1).to(x_current_flatten.device)
        x_pos_key = self.x_pos_key.repeat(batch_size // 2, 1, 1).to(x_current_flatten.device)
        
        # decoder layer
        x = self.temporal_decoder(x_current_flatten, x_history_flatten, x_pos, x_pos_key)

        x = x.view(batch_size // len_queue, c, h, w)
        
        # unpool back to (2, 512, 180, 180)
        x = self.unpool(x, indices_pool_current)
        
        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            if self.use_depth_loss:
                if 'depth' in auxiliary_losses:
                    outputs["loss/depth"] = auxiliary_losses['depth']
                else:
                    raise ValueError('Use depth loss is true, but depth loss not found')
            return outputs
        else:
            outputs = [{} for _ in range(batch_size // len_queue)] # "batch_size" is actucally bs*len_queue, it means range(bs)
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels, token) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                                "sample_token": token,
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs

    def save_tensor(self, tensor, filename='tensor_saved.pt'):
        filename = 'save_tensors/' + filename
        try:
            torch.save(tensor, filename)
            print(f"Tensor saved successfully to {filename}")
        except Exception as e:
            print(f"Error occurred while saving tensor: {e}")
