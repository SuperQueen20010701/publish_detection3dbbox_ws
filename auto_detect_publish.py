'''
需要兼容单目 | 多传感器 | lidar detection 的多范围场景
'''
import argparse
import copy
from argparse import ArgumentParser
import os
import sys
import time
from tkinter import N
import traceback
from pathlib import Path
from typing import Optional ,Sequence,Tuple,Dict,Any,Union,List
import mmcv
import logging
import numpy as np
from unittest import TestCase
from torch import Tensor
import mmengine
import torch
from pathlib import Path
# ros publish
import rospy
from sensor_msgs.msg import PointCloud2,PointField
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
# detection inferencer
try:
    from mmdet3d.apis import (
                                MultiModalityDet3DInferencer, 
                                LidarDet3DInferencer,
                                MonoDet3DInferencer,
                                init_model)
    from mmdet3d.structures import (BaseInstance3DBoxes, DepthInstance3DBoxes,
                                    Det3DDataSample, LiDARInstance3DBoxes,
                                    CameraInstance3DBoxes)
    from mmdet3d.structures.bbox_3d import Box3DMode
    from mmengine.config import Config,ConfigDict 
    from mmengine.dataset import Compose
    from mmdet3d.utils import ConfigType
    from mmengine.visualization.utils import (check_type,tensor2ndarray)
    from mmengine import load
    from mmengine.structures import InstanceData
    from mmengine.fileio import (get_file_backend, isdir, join_path,
                             list_dir_or_file)
    from mmengine.infer.infer import ModelType
    from mmdet3d.visualization import to_depth_mode
    from mmengine.logging import print_log
    from mmdet3d.registry import VISUALIZERS
except ImportError as e:
    print(f"Error importing mmdet3d: {e}")
    print("Please make sure mmdet3d is properly installed.")
    sys.exit(1)

InputsType = Union[str, np.ndarray, Sequence[Union[str, np.ndarray]]]
ConfigType = Union[Config, ConfigDict]
InstanceList = List[InstanceData]
PredType = Union[InstanceData, InstanceList]

def save_pointcloud_ply(points: np.ndarray, filepath: str) -> None:
    """Save point cloud as PLY format.
    
    Args:
        points: Point cloud array of shape (N, 3) or (N, 4) [x, y, z, intensity]
        filepath: Output file path with .ply extension
    """
    if points is None or len(points) == 0:
        return
    
    # Ensure points have at least x, y, z
    if points.shape[1] < 3:
        raise ValueError(f"Points must have at least 3 dimensions (x, y, z), got {points.shape[1]}")
    
    num_points = len(points)
    has_intensity = points.shape[1] >= 4
    
    # Write PLY file
    with open(filepath, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_intensity:
            f.write("property float intensity\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(num_points):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            if has_intensity:
                intensity = points[i, 3]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {intensity:.6f}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
# 参数解析
def parse_args():
    parser = ArgumentParser(description='MVXNet ROS Detection Node publisher')
    parser.add_argument('--pcd_root',dest='pcd_root',help='Point cloud file')
    parser.add_argument('--img_root',dest='img_root',help='image file')
    parser.add_argument('--infos',dest='infos',help='info pickle file')
    parser.add_argument('--model',dest='model',help='Config file')
    parser.add_argument('--weights',dest='weights',help='Checkpoint file')
    parser.add_argument('--img_out_dir',dest='img_out_dir',help='output directory of visualization images')
    parser.add_argument('--device',default='cuda:0',help='Device used for inference')
    parser.add_argument('--cam-type',type=str,default='CAM2',help='choose camera type to inference')
    parser.add_argument('--pred-score-thr',type=float,default=0.3,help='bbox score threshold')
    # ROS pointcloud publish options
    parser.add_argument('--pcd-topic', type=str, default='velodyne_points',
                        help='ROS topic for publishing raw pointcloud')
    parser.add_argument('--filtered-pcd-topic', type=str, default='velodyne_points_filtered',
                        help='ROS topic for publishing filtered pointcloud (points outside predicted boxes, removed points inside boxes)')
    parser.add_argument('--publish-raw-pcd', dest='publish_raw_pcd', action='store_true',
                        help='Enable publishing raw pointcloud')
    parser.set_defaults(publish_raw_pcd=False)  # Default: don't publish raw PCD
    parser.add_argument('--publish-filtered-pcd', dest='publish_filtered_pcd', action='store_true',
                        help='Disable publishing filtered pointcloud (points outside predicted boxes, removed points inside boxes)')
    parser.set_defaults(publish_filtered_pcd=True)  # Default: publish filtered PCD (points outside boxes)
    # runtime flags (keep backward-compatible aliases)
    parser.add_argument('--no-visualize', dest='no_visualize',
                        action='store_true', help='Disable visualization function')
    parser.add_argument('--no-postprocess', dest='no_postprocess',
                        action='store_true', help='Disable postprocess function')
    parser.add_argument('--fp16', action='store_true',help='Enable autocast fp16 inference (CUDA only).')
    parser.add_argument('--show', action='store_true',
                        help='Show visualization windows (requires GUI).')
    parser.add_argument('--vis-save-interval', dest = 'vis_save_interval',type=int, default=1,
                        help='Save visualization every N frames (1=every frame, 10=every 10th frame). Default: 1')
    parser.add_argument('--vis-wait-time',dest = 'vis_wait_time', type=float, default=0.0,
                        help='Wait time for visualization window (seconds). 0.0 for minimal blocking. Default: 0.0')
    call_args = vars(parser.parse_args())

    # If user did not specify img_out_dir explicitly, reuse out_dir.
    if not call_args.get('img_out_dir'):
        call_args['img_out_dir'] = '/kitti_data/fast_result_pred'

    if call_args.get('pcd_root') is None and call_args.get('img_root') is None:
        raise ValueError('pcd_root or img_root is required')

    call_args['inputs'] = dict(
        points = call_args.get('pcd_root',None),
        img = call_args.get('img_root',None),
        infos= call_args.get('infos',None))
    
    # Always remove these keys from call_args, regardless of their values
    # They are now stored in call_args['inputs'] and shouldn't be passed to run_inference
    call_args.pop('pcd_root', None)
    call_args.pop('img_root', None)
    call_args.pop('infos', None)
    # Decide device (and validate fp16)
    cuda_ok = False
    try:
        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    init_args = {}
    init_args['model'] = call_args.pop('model')
    init_args['weights'] = call_args.pop('weights')
    requested_device = call_args.pop('device')

    if not cuda_ok:
        init_args['device'] = 'cpu'
        if call_args.get('fp16'):
            print_log(
                'WARNING: --fp16 requested but CUDA is not available. Disable fp16.',
                logger='current', level=logging.WARNING)
            call_args['fp16'] = False
    else:
        init_args['device'] = requested_device
    return init_args, call_args


class MultiModalityDetectionInferencerNode(MultiModalityDet3DInferencer,
                                LidarDet3DInferencer,
                                MonoDet3DInferencer):
    def __init__(self ,
                 model:Union[ModelType,str,None]=None,
                 weights:Optional[str]=None,
                 device:Optional[str] = None,
                 scope:str = 'mmdet3d',
                 palette:str = 'none',**kwargs):
        # First, load config to determine which modality to use
        # This must be done BEFORE calling base class __init__
        cfg_ = self.load_cfg(model) if model is not None else None
        if cfg_ is None:
            raise ValueError('Failed to load config from model. Please check the model path or config.')
        
        # Check which modalities are used
        use_modality_dict = self._check_use_modality(cfg_)
        self.use_camera = use_modality_dict.get('use_camera', False)
        self.use_lidar = use_modality_dict.get('use_lidar', False)
        
        # Call appropriate base class __init__ based on modality
        if self.use_camera and self.use_lidar:
            MultiModalityDet3DInferencer.__init__(self,
                                                  model=model,
                                                  weights=weights,
                                                  device=device,
                                                  scope=scope,
                                                  palette=palette)
        elif self.use_camera:
            MonoDet3DInferencer.__init__(self,
                                        model=model,
                                        weights=weights,
                                        device=device,
                                        scope=scope,
                                        palette=palette)
        elif self.use_lidar:
            LidarDet3DInferencer.__init__(self,
                                        model=model,
                                        weights=weights,
                                        device=device,
                                        scope=scope,
                                        palette=palette)
        else:
            raise ValueError('No modality is used in the test pipeline')    

        self.pipeline_ = self.pipeline if self.pipeline is not None else print_log(f'DEBUG: Load pipeline failed', logger='current', level=logging.INFO)
        self.use_dim = None
        self.coord_type = None
        self.load_dim = None
        self.cam_type = None

        self.num_of_proc_count = 0
        self.ros_publisher_node = None
        self.ros_enabled = True
        self.makePublishBbox = None
        self.pubBboxMsg = None
        self.makePublishPcd = None
        self.pubPcdMsg = None
        self.ros_topic = 'detection/bboxes'
        self.rate = None
        self.out_pcd_img_path = None

        self.vis_save_interval = 50
        self.vis_wait_time = 0.0
        self.vis_skip_frame = 0
        self.vis_frame_count = 0  # Initialize frame counter
        # Store modality info for later use
        self.use_modality = {'use_camera': self.use_camera, 'use_lidar': self.use_lidar}
        # ROS pointcloud publish behavior (default: keep existing raw publish)
        self.publish_raw_pcd = False
        self.publish_filtered_pcd = True

    def load_cfg(self ,model:Union[ModelType, str, None] = None) -> ConfigType:
        cfg : ConfigType = None
        print_log(f'DEBUG!! model {type(model)}', logger='current', level=logging.INFO)
        if isinstance(model , str):
            if os.path.isfile(model):
                cfg = Config.fromfile(model)
                print_log(f'DEBUG!! model config {type(cfg)}', logger='current', level=logging.INFO)
            else:
                cfg ,_ = self._load_model_from_metafile(model)
        elif isinstance(model , (Config , ConfigDict)):
            cfg = copy.deepcopy(model)
        elif isinstance(model , dict):
            cfg = copy.deepcopy(ConfigDict(model))
        elif model is None:
            cfg = ConfigDict()
        else:
            raise TypeError('model must be a filepath or any ConfigType'
                        f'object, but got {type(model)}')
        return cfg

    def _check_use_modality(self, cfg: ConfigType = None) -> Dict[str, bool]:
        """Check if the model is multi-modality (lidar + camera) or single modality.
        
        Args:
            cfg: Config object to check modality from.
            
        Returns:
            Dict[str, bool]: Dictionary with 'use_camera' and 'use_lidar' keys.
        """
        use_camera_ = False
        use_lidar_ = False
        
        if cfg is not None:
            try:
                modality_cfg = cfg.test_dataloader.dataset.modality
                if isinstance(modality_cfg, dict):
                    use_camera_ = modality_cfg.get('use_camera', False)
                    use_lidar_ = modality_cfg.get('use_lidar', False)
            except (AttributeError, KeyError) as e:
                print_log(f'WARNING: Failed to get modality from config: {e}. '
                         f'Assuming use_camera=False, use_lidar=False',
                         logger='current', level=logging.WARNING)
        
        return {'use_camera': use_camera_, 'use_lidar': use_lidar_} 
        
    def _init_pipeline(self, cfg: ConfigType):
        """Initialize pipeline based on modality."""
        if self.use_camera and self.use_lidar:
            return MultiModalityDet3DInferencer._init_pipeline(self, cfg)
        elif self.use_camera:
            return MonoDet3DInferencer._init_pipeline(self, cfg)
        elif self.use_lidar:
            return LidarDet3DInferencer._init_pipeline(self, cfg)
        else:
            raise ValueError('No modality is used in the test pipeline')   

    def load_param_from_cfg(self) ->None:
        if self.cfg is None:
            print_log('ERROR !! cfg is None',logger='current',level=logging.WARNING)
            return 
        try:
            # Try to get pipeline from test_dataloader.dataset.pipeline first
            pipeline_cfg = None
            if hasattr(self.cfg, 'test_dataloader') and hasattr(self.cfg.test_dataloader, 'dataset'):
                pipeline_cfg = self.cfg.test_dataloader.dataset.pipeline
            # If not found, try test_pipeline
            if pipeline_cfg is None and hasattr(self.cfg, 'test_pipeline'):
                pipeline_cfg = self.cfg.test_pipeline
                print_log('INFO: Using test_pipeline instead of test_dataloader.dataset.pipeline', 
                         logger='current', level=logging.INFO)
            
            if pipeline_cfg is None:
                print_log('WARNING: No pipeline found in config, using default values', 
                         logger='current', level=logging.WARNING)
                self.coord_type = 'LIDAR'
                self.load_dim = 4
                self.use_dim = list(range(4))
                return
                
            if self.use_lidar :
                load_point_idx = self._get_transform_idx(pipeline_cfg,'LoadPointsFromFile')
                if load_point_idx == -1:
                    print_log(f'WARNING: LoadPointsFromFile not found in pipeline (length={len(pipeline_cfg)}), using default values', 
                            logger='current', level=logging.WARNING)
                    # Log pipeline structure for debugging
                    pipeline_types = [t.get('type', 'unknown') for t in pipeline_cfg]
                    print_log(f'DEBUG: Pipeline types: {pipeline_types}', 
                             logger='current', level=logging.DEBUG)
                    # Set default values when LoadPointsFromFile is not found
                    self.coord_type = 'LIDAR'
                    self.load_dim = 4  # Default: x, y, z, intensity
                    self.use_dim = list(range(4))  # Default: use all 4 dimensions
                else:
                    load_cfg = pipeline_cfg[load_point_idx]
                    self.coord_type = load_cfg.get('coord_type', 'LIDAR')
                    self.load_dim = load_cfg.get('load_dim', 4)
                    self.use_dim = load_cfg.get('use_dim', 4)
                    if isinstance(self.use_dim, int):
                        self.use_dim = list(range(self.use_dim))
                    print_log(f'INFO: Loaded params from LoadPointsFromFile: coord_type={self.coord_type}, load_dim={self.load_dim}, use_dim={self.use_dim}',
                             logger='current', level=logging.INFO)
            elif self.use_camera and self.use_lidar == False:
                print_log(f'INFO !! Only for mono detection using camera only')
        except Exception as e:
            print_log(f'ERROR !! Failed to get parameters from pipeline: {e}',logger='current',level=logging.ERROR)
            import traceback
            print_log(traceback.format_exc(), logger='current', level=logging.DEBUG)
            # Set default values on error
            self.coord_type = 'LIDAR'
            self.load_dim = 4
            self.use_dim = list(range(4))
            return
        print_log(f'DEBUG: Final params - coord_type={self.coord_type}, load_dim={self.load_dim}, use_dim={self.use_dim}',
                 logger='current', level=logging.INFO)

    def _init_model_pipeline(self,
                             out_pcd_img_path: Optional[str] = None,
                             cam_type: Optional[str] = None,
                             pred_score_thr: float = 0.3,
                             pcd_topic: str = 'velodyne_points',
                             filtered_pcd_topic: str = 'velodyne_points_filtered',
                             publish_raw_pcd: bool = False,
                             publish_filtered_pcd: bool = True,
                             **kwargs) -> None:
        if self.model is not  None and self.cfg is None:
            self.cfg = self.load_cfg(self.model)
        else:
            print_log(f'DEBUG: Config and Model load successfully ', logger='current', level=logging.INFO)

        self.load_param_from_cfg()
        self.publish_raw_pcd = bool(publish_raw_pcd)
        self.publish_filtered_pcd = bool(publish_filtered_pcd)
        if self.ros_enabled and self.ros_topic is not None:
            self.ros_publisher_node = Det3DRosPublishNode(
                topic=self.ros_topic,
                frame_id='velodyne',
                pred_score_thr=float(pred_score_thr),
                ros_node_name='detection_bbox_publisher',
                queue_size=10,
                latch=True,
                enabled_ros=self.ros_enabled,
                pcd_topic=str(pcd_topic),
                filtered_pcd_topic=str(filtered_pcd_topic)
            )

        if self.ros_publisher_node is not None:
            self.makePublishBbox = self.ros_publisher_node.make_publish_bbox
            self.pubBboxMsg = self.ros_publisher_node.publish_bbox  # Fix: should be publish_bbox, not make_publish_bbox
            self.pubPcdMsg = self.ros_publisher_node.publish_pcd
            self.pubFilteredPcdMsg = self.ros_publisher_node.publish_filtered_pcd
            self.rate = self.ros_publisher_node._rate
        if out_pcd_img_path is not None:
            self.out_pcd_img_path = out_pcd_img_path
        else:
            self.out_pcd_img_path = kwargs.pop('img_out_dir', None)
        
        if cam_type is not None:
            self.cam_type = cam_type
        else:
            self.cam_type = 'CAM2'

    def _filter_points_in_pred_boxes(self,
                                    pcd: np.ndarray,
                                    data_sample: Det3DDataSample,
                                    score_thr: float,
                                    lidar2cam: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return points OUTSIDE predicted 3D boxes (remove points inside boxes, LiDAR coord only).
        
        Args:
            pcd: Point cloud in LiDAR coordinates
            data_sample: Detection result data sample
            score_thr: Score threshold for filtering boxes
            lidar2cam: Optional lidar2cam transformation matrix (for mono detection)
        """
        print_log(f'DEBUG: _filter_points_in_pred_boxes called - pcd shape={pcd.shape if pcd is not None else None}, data_sample={data_sample is not None}',
                 logger='current', level=logging.INFO)
        
        if pcd is None or data_sample is None:
            print_log('DEBUG: _filter_points_in_pred_boxes: pcd or data_sample is None', 
                      logger='current', level=logging.DEBUG)
            return None,None
        if not hasattr(data_sample, 'pred_instances_3d'):
            print_log('DEBUG: _filter_points_in_pred_boxes: data_sample has no pred_instances_3d', 
                      logger='current', level=logging.DEBUG)
            return None,None

        pred_instances_3d = data_sample.pred_instances_3d
        if pred_instances_3d is None or not hasattr(pred_instances_3d, 'bboxes_3d'):
            print_log('DEBUG: _filter_points_in_pred_boxes: pred_instances_3d is None or has no bboxes_3d', 
                      logger='current', level=logging.DEBUG)
            return None,None

        bboxes_3d = getattr(pred_instances_3d, 'bboxes_3d', None)
        if bboxes_3d is None:
            print_log('WARNING: _filter_points_in_pred_boxes: bboxes_3d is None', 
                      logger='current', level=logging.WARNING)
            return None,None
        
        print_log(f'INFO: _filter_points_in_pred_boxes: bboxes_3d type={type(bboxes_3d).__name__}, len={len(bboxes_3d)}, lidar2cam provided={lidar2cam is not None}',
                 logger='current', level=logging.INFO)

        # Convert CameraInstance3DBoxes to LiDARInstance3DBoxes if needed
        # (points are in velodyne/LiDAR frame, so boxes need to be in LiDAR frame too)
        if isinstance(bboxes_3d, CameraInstance3DBoxes):
            # Try to get lidar2cam from parameter, metainfo, or input
            if lidar2cam is None:
                if hasattr(data_sample, 'metainfo') and data_sample.metainfo is not None:
                    # Try to get from metainfo
                    if 'lidar2cam' in data_sample.metainfo:
                        lidar2cam = data_sample.metainfo['lidar2cam']
            
            # If we have lidar2cam, convert boxes to LiDAR frame
            if lidar2cam is not None:
                try:
                    lidar2cam = np.array(lidar2cam) if not isinstance(lidar2cam, np.ndarray) else lidar2cam
                    # Ensure 4x4 matrix
                    if lidar2cam.shape == (3, 4):
                        lidar2cam_4x4 = np.eye(4, dtype=lidar2cam.dtype)
                        lidar2cam_4x4[:3, :] = lidar2cam
                        lidar2cam = lidar2cam_4x4
                    # Convert to tensor
                    if isinstance(lidar2cam, np.ndarray):
                        lidar2cam = torch.from_numpy(lidar2cam).float()
                    # Convert camera boxes to LiDAR boxes
                    # Note: convert_to expects cam2lidar (inverse of lidar2cam)
                    cam2lidar = torch.inverse(lidar2cam)
                    bboxes_3d = bboxes_3d.convert_to(Box3DMode.LIDAR, cam2lidar, correct_yaw=True)
                    print_log(f'Converted CameraInstance3DBoxes to LiDARInstance3DBoxes for point cloud filtering',
                             logger='current', level=logging.INFO)
                except Exception as e:
                    print_log(
                        f'WARNING: Failed to convert CameraInstance3DBoxes to LiDARInstance3DBoxes: {e}. '
                        'Skip filtering.',
                        logger='current', level=logging.WARNING)
                    import traceback
                    print_log(traceback.format_exc(), logger='current', level=logging.DEBUG)
                    return None, None
            else:
                print_log(
                    f'WARNING: Filtered PCD publish requested, but bboxes_3d is CameraInstance3DBoxes '
                    'and lidar2cam not found. Skip filtering.',
                    logger='current', level=logging.WARNING)
                return None, None
        elif not isinstance(bboxes_3d, LiDARInstance3DBoxes):
            print_log(
                f'WARNING: Filtered PCD publish requested, but bboxes_3d type is {type(bboxes_3d)}. '
                'Skip filtering (expects LiDARInstance3DBoxes or CameraInstance3DBoxes).',
                logger='current', level=logging.WARNING)
            return None, None

        # Score filtering (keep consistent with bbox publish threshold)
        num_boxes_before = len(bboxes_3d)
        if hasattr(pred_instances_3d, 'scores_3d'):
            scores_3d = pred_instances_3d.scores_3d
            keep_box = scores_3d > float(score_thr)
            try:
                bboxes_3d = bboxes_3d[keep_box]
            except Exception:
                # Fallback: if indexing fails for some structure variant, skip score filtering.
                pass
        num_boxes_after = len(bboxes_3d)
        print_log(f'DEBUG: Filtering with {num_boxes_before} boxes (after score_thr={score_thr}: {num_boxes_after} boxes)', 
                  logger='current', level=logging.DEBUG)

        if bboxes_3d.tensor.numel() == 0:
            print_log('DEBUG: _filter_points_in_pred_boxes: No boxes after filtering', 
                      logger='current', level=logging.DEBUG)
            return pcd[:0].copy(),pcd[:0].copy()

        # Log box information for debugging
        bboxes_tensor = bboxes_3d.tensor.cpu().numpy()
        for i, box in enumerate(bboxes_tensor):
            center = box[:3]
            dims = box[3:6]
            yaw = box[6]
            print_log(f'INFO: Box {i+1}: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), '
                     f'dims=({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}), yaw={yaw:.2f}',
                     logger='current', level=logging.INFO)

        # Compute points-in-box mask
        pts_xyz = torch.from_numpy(pcd[:, :3]).to(device=bboxes_3d.tensor.device, dtype=torch.float32)
        in_any = bboxes_3d.points_in_boxes_all(pts_xyz).to(torch.bool).any(dim=1)
        in_any_np = in_any.detach().cpu().numpy()
        
        # Invert mask to get points OUTSIDE boxes (remove points inside boxes)
        out_any_np = ~in_any_np
        
        num_points_inside = in_any_np.sum()
        num_points_outside = out_any_np.sum()
        num_points_total = len(pcd)
        filter_ratio_inside = (num_points_inside / num_points_total * 100) if num_points_total > 0 else 0.0
        filter_ratio_outside = (num_points_outside / num_points_total * 100) if num_points_total > 0 else 0.0
        
        # Count points per box
        points_per_box = bboxes_3d.points_in_boxes_all(pts_xyz).to(torch.bool).sum(dim=0).cpu().numpy()
        for i, count in enumerate(points_per_box):
            print_log(f'INFO: Box {i+1} contains {count} points (will be removed)', 
                     logger='current', level=logging.INFO)
        
        print_log(f'INFO: Removing {num_points_inside} points inside {num_boxes_after} boxes '
                 f'({filter_ratio_inside:.2f}%), keeping {num_points_outside} points outside '
                 f'({filter_ratio_outside:.2f}% out of {num_points_total} total points)', 
                 logger='current', level=logging.INFO)
        return pcd[out_any_np],pcd[in_any_np]

    def _inputs_to_list(self,
                        inputs: Union[dict, list],
                        cam_type: Optional[str] = 'CAM2',
                        **kwargs) -> list:
        if self.use_camera and self.use_lidar:
            return MultiModalityDet3DInferencer._inputs_to_list(self, inputs, cam_type, **kwargs)
        elif self.use_camera:
            # For mono detection, use improved version that handles filename matching more flexibly
            return self._inputs_to_list_mono(inputs, cam_type, **kwargs)
        elif self.use_lidar:
            return LidarDet3DInferencer._inputs_to_list(self, inputs, **kwargs)
        else:
            raise ValueError('No modality is used in the test pipeline')
    
    def _inputs_to_list_mono(self,
                             inputs: Union[dict, list],
                             cam_type: Optional[str] = 'CAM2',
                             **kwargs) -> list:
        """Improved _inputs_to_list for mono detection with flexible filename matching.
        
        This method is more flexible than the base MonoDet3DInferencer version:
        - Allows filename-only matching (ignores path differences)
        - Provides better error messages
        - Handles cases where infos might not perfectly match
        """
        from mmengine.fileio import get_file_backend, isdir, join_path, list_dir_or_file
        import mmengine
        import os.path as osp
        
        if isinstance(inputs, dict):
            if 'infos' not in inputs:
                raise ValueError('infos is required for mono detection. Please provide --infos argument.')
            infos = inputs.pop('infos')

            if isinstance(inputs.get('img'), str):
                img = inputs['img']
                backend = get_file_backend(img)
                if hasattr(backend, 'isdir') and isdir(img):
                    filename_list = list_dir_or_file(img, list_dir=False)
                    inputs = [{
                        'img': join_path(img, filename)
                    } for filename in filename_list]

            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]

            # Load infos
            loaded_infos = mmengine.load(infos)
            if isinstance(loaded_infos, dict):
                if 'data_list' not in loaded_infos:
                    raise ValueError('Invalid infos file format: missing "data_list" key')
                info_list = loaded_infos['data_list']
            elif isinstance(loaded_infos, (list, tuple)):
                info_list = list(loaded_infos)
            else:
                raise TypeError(f'Invalid infos file format: expected dict or list, got {type(loaded_infos)}')
            
            if len(info_list) != len(inputs):
                print_log(f'WARNING: Number of inputs ({len(inputs)}) does not match info_list length ({len(info_list)}). '
                         f'Will try to match by filename.', logger='current', level=logging.WARNING)

            # Match inputs with info_list by filename
            matched_inputs = []
            for input_item in inputs:
                img_path = input_item.get('img')
                if not isinstance(img_path, str):
                    # If img is not a string, use index-based matching
                    if len(matched_inputs) < len(info_list):
                        data_info = info_list[len(matched_inputs)]
                    else:
                        raise ValueError(f'Cannot match input: no corresponding info entry available')
                else:
                    img_basename = osp.basename(img_path)
                    # Try to find matching info entry
                    data_info = None
                    for info_item in info_list:
                        if cam_type in info_item.get('images', {}):
                            info_img_path = info_item['images'][cam_type].get('img_path', '')
                            info_img_basename = osp.basename(info_img_path)
                            if img_basename == info_img_basename:
                                data_info = info_item
                                break
                    
                    # If no exact match, try index-based matching
                    if data_info is None:
                        idx = len(matched_inputs)
                        if idx < len(info_list):
                            data_info = info_list[idx]
                            print_log(f'WARNING: No exact filename match for {img_basename}, using info entry at index {idx}',
                                     logger='current', level=logging.WARNING)
                        else:
                            raise ValueError(f'Cannot find matching info entry for image {img_basename}. '
                                           f'Available info entries: {len(info_list)}')
                
                # Extract camera parameters
                if cam_type not in data_info.get('images', {}):
                    raise ValueError(f'Camera type {cam_type} not found in info file. '
                                   f'Available cameras: {list(data_info.get("images", {}).keys())}')
                
                cam_info = data_info['images'][cam_type]
                cam2img = np.asarray(cam_info['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(cam_info.get('lidar2cam', np.eye(4)[:3, :]), dtype=np.float32)
                
                if 'lidar2img' in cam_info:
                    lidar2img = np.asarray(cam_info['lidar2img'], dtype=np.float32)
                else:
                    # Compute lidar2img from cam2img and lidar2cam
                    if lidar2cam.shape == (3, 4):
                        lidar2cam_4x4 = np.eye(4, dtype=np.float32)
                        lidar2cam_4x4[:3, :] = lidar2cam
                        lidar2cam = lidar2cam_4x4
                    if cam2img.shape == (3, 3):
                        cam2img_4x4 = np.eye(4, dtype=np.float32)
                        cam2img_4x4[:3, :3] = cam2img
                        cam2img = cam2img_4x4
                    elif cam2img.shape == (3, 4):
                        cam2img_4x4 = np.eye(4, dtype=np.float32)
                        cam2img_4x4[:3, :] = cam2img
                        cam2img = cam2img_4x4
                    lidar2img = cam2img @ lidar2cam
                
                input_item['cam2img'] = cam2img
                input_item['lidar2cam'] = lidar2cam
                input_item['lidar2img'] = lidar2img
                matched_inputs.append(input_item)
            
            return matched_inputs
            
        elif isinstance(inputs, (list, tuple)):
            # Handle list of inputs
            result = []
            for input_item in inputs:
                if 'infos' not in input_item:
                    raise ValueError('Each input item must contain "infos" for mono detection')
                infos = input_item.pop('infos')
                info_list = mmengine.load(infos)
                
                if isinstance(info_list, dict):
                    if 'data_list' not in info_list:
                        raise ValueError('Invalid infos file format: missing "data_list" key')
                    info_list = info_list['data_list']
                elif isinstance(info_list, (list, tuple)):
                    info_list = list(info_list)
                else:
                    raise TypeError(f'Invalid infos file format: expected dict or list, got {type(info_list)}')
                
                if len(info_list) != 1:
                    raise ValueError(f'Only support single sample info in `.pkl`, got {len(info_list)} entries')
                
                data_info = info_list[0]
                if cam_type not in data_info.get('images', {}):
                    raise ValueError(f'Camera type {cam_type} not found in info file')
                
                cam_info = data_info['images'][cam_type]
                cam2img = np.asarray(cam_info['cam2img'], dtype=np.float32)
                lidar2cam = np.asarray(cam_info.get('lidar2cam', np.eye(4)[:3, :]), dtype=np.float32)
                
                if 'lidar2img' in cam_info:
                    lidar2img = np.asarray(cam_info['lidar2img'], dtype=np.float32)
                else:
                    # Compute lidar2img
                    if lidar2cam.shape == (3, 4):
                        lidar2cam_4x4 = np.eye(4, dtype=np.float32)
                        lidar2cam_4x4[:3, :] = lidar2cam
                        lidar2cam = lidar2cam_4x4
                    if cam2img.shape == (3, 3):
                        cam2img_4x4 = np.eye(4, dtype=np.float32)
                        cam2img_4x4[:3, :3] = cam2img
                        cam2img = cam2img_4x4
                    elif cam2img.shape == (3, 4):
                        cam2img_4x4 = np.eye(4, dtype=np.float32)
                        cam2img_4x4[:3, :] = cam2img
                        cam2img = cam2img_4x4
                    lidar2img = cam2img @ lidar2cam
                
                input_item['cam2img'] = cam2img
                input_item['lidar2cam'] = lidar2cam
                input_item['lidar2img'] = lidar2img
                result.append(input_item)
            
            return result
        
        return list(inputs) if not isinstance(inputs, (list, tuple)) else inputs

    def visualize(self,
                  inputs: InputsType,
                  preds: PredType,
                  return_vis: bool = False,
                  show: bool = False,
                  wait_time: int = 0,
                  draw_pred: bool = True,
                  pred_score_thr: float = 0.3,
                  no_save_vis: bool = False,
                  img_out_dir: str = '',
                  cam_type_dir: str = 'CAM2') -> Union[List[np.ndarray], None]:
        """Override visualize to route to correct inferencer based on modality.
        
        This method routes visualization calls to the appropriate inferencer
        based on which modalities are being used, avoiding MRO issues.
        """
        # Route to correct inferencer based on modality
        if self.use_lidar and not self.use_camera:
            # Pure LiDAR: use LidarDet3DInferencer.visualize()
            # Remove cam_type_dir from kwargs as LidarDet3DInferencer.visualize() doesn't accept it
            kwargs = {
                'return_vis': return_vis,
                'show': show,
                'wait_time': wait_time,
                'draw_pred': draw_pred,
                'pred_score_thr': pred_score_thr,
                'no_save_vis': no_save_vis,
                'img_out_dir': img_out_dir
            }
            return LidarDet3DInferencer.visualize(
                self, inputs, preds, **kwargs)
        elif self.use_camera and not self.use_lidar:
            # Mono detection: use MonoDet3DInferencer.visualize()
            kwargs = {
                'return_vis': return_vis,
                'show': show,
                'wait_time': wait_time,
                'draw_pred': draw_pred,
                'pred_score_thr': pred_score_thr,
                'no_save_vis': no_save_vis,
                'img_out_dir': img_out_dir,
                'cam_type_dir': cam_type_dir
            }
            return MonoDet3DInferencer.visualize(
                self, inputs, preds, **kwargs)
        else:
            # Multi-modality: use MultiModalityDet3DInferencer.visualize()
            kwargs = {
                'return_vis': return_vis,
                'show': show,
                'wait_time': wait_time,
                'draw_pred': draw_pred,
                'pred_score_thr': pred_score_thr,
                'no_save_vis': no_save_vis,
                'img_out_dir': img_out_dir,
                'cam_type_dir': cam_type_dir
            }
            return MultiModalityDet3DInferencer.visualize(
                self, inputs, preds, **kwargs)

    def run_inference(self, inputs: InputsType, batch_size: int = 1,
                     return_datasample: bool = False, **kwargs) -> Optional[dict]:

        no_visualize = bool(kwargs.pop('no_visualize', kwargs.pop('no-visualize', False)))
        no_postprocess = bool(kwargs.pop('no_postprocess', kwargs.pop('no-postprocess', False)))
        fp16 = bool(kwargs.pop('fp16', False))
        pred_score_thr = float(kwargs.pop('pred_score_thr', 0.3))

        show = bool(kwargs.pop('show', False))
        fp16_enabled = fp16

        vis_save_interval = int(kwargs.pop('vis_save_interval',self.vis_save_interval))
        vis_wait_time = float(kwargs.pop('vis_wait_time',self.vis_wait_time))
        # 处理传入解析参数
        (preprocess_kwargs,
        forward_kwargs,
        visualize_kwargs,
        postprocess_kwargs) = self._dispatch_kwargs(**kwargs)
        # 相机处理类型
        camera_type = preprocess_kwargs.pop('cam_type','CAM2')
        # Ensure visualize saves to the expected camera folder name
        if 'cam_type_dir' not in visualize_kwargs:
            visualize_kwargs['cam_type_dir'] = camera_type

        # origin input 
        origin_inputs = self._inputs_to_list(inputs,cam_type=camera_type)
        if not isinstance(origin_inputs, list):
            origin_inputs = list(origin_inputs)
        detection_inputs = self.preprocess(origin_inputs,batch_size=batch_size,**preprocess_kwargs)
        results_dict = {'predictions': [], 'visualization': []}
        for idx, single_data in enumerate(detection_inputs):
            if single_data is None:
                print_log(f'ERROR !! failed to get single input data',logger='current',level=logging.INFO)
                continue
            self.num_of_proc_count = idx + 1
            with torch.no_grad():
                if fp16_enabled and torch.cuda.is_available():
                    try:
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            pred = self.forward(single_data, **forward_kwargs)
                    except RuntimeError as e:
                            print_log(
                                'WARNING: fp16 requested but MMCV does not support fp16 )',
                                logger='current', level=logging.WARNING)
                            fp16_enabled = False
                            pred = self.forward(single_data, **forward_kwargs)
                else:
                    pred = self.forward(single_data, **forward_kwargs)

            # Normalize pred to a single Det3DDataSample for this frame.
            if isinstance(pred, (list, tuple)):
                if len(pred) == 0:
                    print_log('WARNING: empty pred list for this frame',
                              logger='current', level=logging.WARNING)
                    continue
                if len(pred) > 1:
                    print_log(f'WARNING: pred list length={len(pred)}; use pred[0]',
                              logger='current', level=logging.WARNING)
                pred = pred[0]

            current_origin_input = origin_inputs[idx] if idx < len(origin_inputs) else None
            visualization = None
            
            # For mono detection, automatically infer point cloud path from image path
            # This allows point cloud filtering/denoising even in mono detection mode
            if self.use_camera and not self.use_lidar and current_origin_input is not None:
                if isinstance(current_origin_input, dict):
                    img_path = current_origin_input.get('img')
                    if img_path and 'points' not in current_origin_input:
                        # Try to infer point cloud path from image path (KITTI format)
                        if isinstance(img_path, str):
                            # Try KITTI format: image_2 -> velodyne
                            if 'image_2' in img_path:
                                pcd_path = img_path.replace('image_2', 'velodyne').replace('.png', '.bin').replace('.jpg', '.bin')
                                if os.path.exists(pcd_path):
                                    current_origin_input['points'] = pcd_path
                                    print_log(f'Inferred point cloud path for noise removal: {pcd_path}',
                                             logger='current', level=logging.INFO)
                                else:
                                    # Try alternative: replace image directory with velodyne
                                    img_dir = os.path.dirname(img_path)
                                    img_basename = os.path.basename(img_path)
                                    pcd_basename = img_basename.replace('.png', '.bin').replace('.jpg', '.bin')
                                    # Try to find velodyne directory at same level as image directory
                                    parent_dir = os.path.dirname(img_dir)
                                    velodyne_dir = os.path.join(parent_dir, 'velodyne')
                                    if not os.path.exists(velodyne_dir):
                                        # Try velodyne_reduced
                                        velodyne_dir = os.path.join(parent_dir, 'velodyne_reduced')
                                    if os.path.exists(velodyne_dir):
                                        pcd_path = os.path.join(velodyne_dir, pcd_basename)
                                        if os.path.exists(pcd_path):
                                            current_origin_input['points'] = pcd_path
                                            print_log(f'Inferred point cloud path for noise removal: {pcd_path}',
                                                     logger='current', level=logging.INFO)
            
            # Prepare output directory for saving images
            img_out_dir = self.out_pcd_img_path or '/kitti_data/fast_result_pred'
            cam_dir = camera_type
            if img_out_dir:
                os.makedirs(os.path.join(img_out_dir, 'vis_camera', cam_dir), exist_ok=True)
                os.makedirs(os.path.join(img_out_dir,'filtered_pcd'),exist_ok = True)
                os.makedirs(os.path.join(img_out_dir,'no_filter_pcd'),exist_ok=True)
                os.makedirs(os.path.join(img_out_dir,'raw_pcd'),exist_ok = True)
            # Handle visualization and/or saving pred to image
            if current_origin_input is not None:
                should_save_vis = (self.vis_frame_count % vis_save_interval == 0)
                self.vis_frame_count += 1
                
                if not no_visualize and should_save_vis:
                    # Full visualization mode: save visualization images
                    visualize_kwargs['img_out_dir'] = img_out_dir
                    visualize_kwargs['wait_time'] = vis_wait_time
                    visualize_kwargs['show'] = show
                    try:
                        # Use the overridden visualize() method which routes to correct inferencer
                        visualization = self.visualize(
                            [current_origin_input], [pred], **visualize_kwargs)
                    except AttributeError as e:
                        if 'convert_to_pinhole_camera_parameters' in str(e):
                            print_log(
                                'WARNING: Open3D show mode failed (likely headless environment). '
                                'Fallback to show=False and continue saving images.',
                                logger='current', level=logging.WARNING)
                            visualization = self.visualize(
                                [current_origin_input], [pred], show=False, **visualize_kwargs)
                        else:
                            raise
                elif no_visualize: # for no visualization occassion
                    # No visualization mode: only save pred to image (draw pred on image)
                    visualize_kwargs['img_out_dir'] = img_out_dir
                    visualize_kwargs['wait_time'] = 0.0
                    visualize_kwargs['show'] = False
                    visualize_kwargs['no_save_vis'] = False 
                    try:
                        # Use the overridden visualize() method which routes to correct inferencer
                        visualization = self.visualize(
                            [current_origin_input], [pred], **visualize_kwargs)
                        print_log(f'INFO !! Saved pred to image (no visualization mode)', 
                                 logger='current', level=logging.INFO)
                    except Exception as e:
                        print_log(f'ERROR !! Failed to save pred to image: {e}', 
                                 logger='current', level=logging.WARNING)
            
            # Align pointcloud timestamp with bbox timestamp for this frame.
            if self.ros_publisher_node is not None:
                try:
                    self.ros_publisher_node.publish_time = self.ros_publisher_node.get_bbox_time_stamp(pred)
                except Exception:
                    pass

            # Publish PCD data
            if current_origin_input is not None:
                try:
                    # this need to make sure the pcd data according to the image data to filter and publish 
                    # if is image detection
                    if self.use_lidar == False and self.use_camera:
                        imgs_data = current_origin_input.get('img') if isinstance(current_origin_input, dict) else None
                        if isinstance(imgs_data,str) and 'image_2' in imgs_data.lower():
                            # Replace image_2 with velodyne and change extension to .bin
                            pcd_path = imgs_data.replace('image_2', 'velodyne')
                            # Replace file extension from .png or .jpg to .bin
                            base, ext = os.path.splitext(pcd_path)
                            if ext.lower() in ['.png', '.jpg', '.jpeg']:
                                pcd_path = base + '.bin'
                    else:
                        pcd_path = current_origin_input.get('points') if isinstance(current_origin_input, dict) else None
                    pcd = None
                    if  isinstance(pcd_path, str):
                        pcd_path = str(Path(pcd_path).resolve())
                        if 'velodyne_reduced' in pcd_path.lower():
                            pcd_path = pcd_path.replace("velodyne_reduced", "velodyne")
                        # Ensure the path ends with .bin
                        if not pcd_path.endswith('.bin'):
                            base, ext = os.path.splitext(pcd_path)
                            pcd_path = base + '.bin'
                        if os.path.isfile(pcd_path) and pcd_path.endswith('.bin'):
                            pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
                            pcd_raw_path = os.path.join(img_out_dir, 'raw_pcd', f'{idx:06d}.ply')
                            save_pointcloud_ply(pcd, pcd_raw_path)
                            print_log(f'INFO !! raw pcd saved as PLY: {pcd_raw_path}',
                                          logger='current', level=logging.INFO)
                        else:
                            print_log(f'WARNING !! pcd file not found: {pcd_path}', 
                                     logger='current', level=logging.WARNING)
                    elif isinstance(pcd_path, np.ndarray):
                        # Points are already loaded as numpy array
                        pcd = pcd_path[:, :4] if(len(pcd_path.shape) == 2 and pcd_path.shape[1] >= 4) else pcd_path
                    else:
                        print_log(f'WARNING !! Unsupported points type: {type(pcd_path)}', 
                                 logger='current', level=logging.WARNING)

                    # Publish raw pointcloud (backward-compatible default)
                    publish_raw = getattr(self, 'publish_raw_pcd', False)
                    publish_filtered = getattr(self, 'publish_filtered_pcd', True)
                    print_log(f'DEBUG: Point cloud publish settings - raw={publish_raw}, filtered={publish_filtered}, pcd shape={pcd.shape if pcd is not None else None}',
                             logger='current', level=logging.DEBUG)
                    
                    if pcd is not None and publish_raw:
                        self.pubPcdMsg(pcd)
                        print_log(f'INFO !! raw pcd published (shape: {pcd.shape})',
                                  logger='current', level=logging.INFO)
                    # Publish filtered pointcloud (points OUTSIDE predicted boxes, removed points inside boxes)
                    elif pcd is not None and publish_filtered:
                        print_log(f'DEBUG: Starting point cloud filtering...',
                                 logger='current', level=logging.DEBUG)
                        # Get lidar2cam from current_origin_input if available (for mono detection)
                        lidar2cam = None
                        if current_origin_input is not None and isinstance(current_origin_input, dict):
                            if 'lidar2cam' in current_origin_input:
                                lidar2cam = current_origin_input['lidar2cam']
                                print_log(f'DEBUG: Found lidar2cam in current_origin_input',
                                         logger='current', level=logging.DEBUG)
                            elif 'metainfo' in current_origin_input and current_origin_input['metainfo'] is not None:
                                if 'lidar2cam' in current_origin_input['metainfo']:
                                    lidar2cam = current_origin_input['metainfo']['lidar2cam']
                                    print_log(f'DEBUG: Found lidar2cam in current_origin_input.metainfo',
                                             logger='current', level=logging.DEBUG)
                        
                        if lidar2cam is None:
                            print_log(f'WARNING: lidar2cam not found in current_origin_input, will try to get from data_sample.metainfo',
                                     logger='current', level=logging.WARNING)
                        
                        filtered ,filter_in_bbox= self._filter_points_in_pred_boxes(
                            pcd=pcd, data_sample=pred, score_thr=pred_score_thr, lidar2cam=lidar2cam)
                        print_log(f'INFO: _filter_points_in_pred_boxes returned - filtered shape={filtered.shape if filtered is not None else None}, filter_in_bbox shape={filter_in_bbox.shape if filter_in_bbox is not None else None}',
                                 logger='current', level=logging.INFO)
                        if filtered is not None:
                            if len(filtered) > 0:
                                # Fix: publish_filtered_pcd only accepts one parameter (pcd)
                                self.pubFilteredPcdMsg(filtered)
                                print_log(f'INFO !! filtered pcd published (shape: {filtered.shape}, removed {len(filter_in_bbox) if filter_in_bbox is not None else 0} points inside boxes)',
                                         logger='current', level=logging.INFO)
                                # save filtered pcd to file
                                if filter_in_bbox is not None and len(filter_in_bbox) > 0:
                                    filtered_pcd_path = os.path.join(img_out_dir, 'filtered_pcd', f'{idx:06d}.bin')
                                    filtered.astype(np.float32).tofile(filtered_pcd_path)
                                    # save filtered pcd as PLY format in the same path
                                    filtered_ply_path = os.path.join(img_out_dir, 'filtered_pcd', f'{idx:06d}.ply')
                                    save_pointcloud_ply(filtered, filtered_ply_path)
                                    print_log(f'INFO !! filtered pcd saved as PLY: {filtered_ply_path}',
                                                logger='current', level=logging.INFO)
                                else:
                                    # No points inside boxes, save to no_filter_pcd directory
                                    if filter_in_bbox is not None and len(filter_in_bbox) > 0:
                                        filter_pcd_path = os.path.join(img_out_dir, 'no_filter_pcd', f'{idx:06d}.bin')
                                        filter_in_bbox.astype(np.float32).tofile(filter_pcd_path)
                            else:
                                print_log(f'WARNING !! filtered pcd is empty (no points outside boxes)',
                                          logger='current', level=logging.WARNING)
                        else:
                            print_log(f'WARNING !! filtered pcd is None (filtering failed or no boxes detected)',
                                      logger='current', level=logging.WARNING)
                    else:
                        print_log(f'WARNING !! pcd is None (filtering failed)',
                                  logger='current', level=logging.WARNING)
                except Exception as e:
                    print_log(f'ERROR !! Failed to publish pcd: {e}', 
                             logger='current', level=logging.WARNING)
                    import traceback
                    print_log(traceback.format_exc(), logger='current', level=logging.DEBUG)
            # Publish ROS bboxes for this frame
            if self.ros_publisher_node is not None:
                self.ros_publisher_node.pred_score_thr = float(pred_score_thr)
            pub_bboxes = self.makePublishBbox(pred)
            if pub_bboxes is not None:
                self.pubBboxMsg(pub_bboxes)
            # Postprocess (optional)
            if not no_postprocess:
                results = self.postprocess([pred], visualization,
                                        return_datasample=return_datasample,
                                        **postprocess_kwargs)
                predictions = results['predictions']
                if hasattr(predictions, '__iter__') and not isinstance(predictions, (list, tuple)):
                    predictions = list(predictions)
                results_dict['predictions'].extend(predictions)
                if results['visualization'] is not None:
                    results_dict['visualization'].extend(results['visualization'])
        return results_dict

class Det3DRosPublishNode:
    def __init__(self,
                topic:str,
                frame_id:str,
                pred_score_thr:float = 0.3,
                ros_node_name:str = 'detection_bbox_publisher',
                queue_size:int = 10,
                latch:bool = True,
                enabled_ros:bool = True,
                publish_rate: float = 10.0,
                pcd_topic: str = 'velodyne_points',
                filtered_pcd_topic: str = 'velodyne_points_filtered'):
        self.topic = topic
        self.frame_id = frame_id
        self.pred_score_thr = float(pred_score_thr)

        self.enabled_ros = bool(enabled_ros)
        self._ros_node_ready = False

        self._rospy = None
        self.ros_node_name = ros_node_name
        self.queue_size = int(queue_size)
        self.latch = bool(latch)
        self.publish_time = None
        self.pcd_topic = 'velodyne_points'

        self._BoundingBox_ = None
        self._BoundingBoxArray_ = None
        self._quaternion_from_euler_ = None
        self._publisher_bbox = None
        self._publisher_pcd = None
        self._publisher_filtered_pcd = None
        self._publish_rate = publish_rate
        self._rate = None
        # initialize ros publisher node 
        self.pcd_topic = str(pcd_topic)
        self.filtered_pcd_topic = str(filtered_pcd_topic)
        enable_result = self._enable_publish_ros()
        print_log(f'ROS publisher initialized:{enable_result}',logger='current',level=logging.INFO)

    def _enable_publish_ros(self) -> bool:
        if not self.enabled_ros:
            return False
        if self._ros_node_ready:
            return True
        
        if not rospy.core.is_initialized():
            try:
                rospy.init_node(self.ros_node_name, anonymous=True, disable_signals=True)

            except Exception as e:
                print_log(f"Failed to initialize ROS node: {e}",logger='current',level=logging.ERROR)
                self.enabled_ros = False
                return False

        # init ros related
        self._rospy = rospy
        self._rate = rospy.Rate(self._publish_rate)
        self._BoundingBox_ = BoundingBox
        self._BoundingBoxArray_ = BoundingBoxArray
        self._quaternion_from_euler_ = quaternion_from_euler
        # bounding box publisher 
        self._publisher_bbox = rospy.Publisher(
            self.topic, BoundingBoxArray, queue_size=self.queue_size, latch=self.latch)
        self._publisher_pcd = rospy.Publisher(
            self.pcd_topic, PointCloud2 ,  queue_size=self.queue_size, latch=self.latch)
        self._publisher_filtered_pcd = rospy.Publisher(
            self.filtered_pcd_topic, PointCloud2, queue_size=self.queue_size, latch=self.latch)
        self._ros_node_ready = True
        print_log(f'ROS publisher has been initialized',logger='current',level=logging.INFO)
        return True

    def get_bbox_time_stamp(self ,data_sample : Det3DDataSample) -> rospy.Time:
        rospy = self._rospy
        try:
            timestamp = data_sample.metainfo.get('timestamp',None)
            if timestamp is not None:
                return rospy.Time.from_sec(float(timestamp))
        except Exception as e:
            print_log(
                f'[Det3DRosPublishHook] get timestamp failed: {e}.',
                logger='current',
            level=logging.WARNING)
        return rospy.Time.now()
    
    def make_publish_msg(self,
                        bboxes_3d : BaseInstance3DBoxes,
                        labels_3d,
                        scores_3d,
                        data_sample : Det3DDataSample = None) -> BoundingBoxArray:
        if bboxes_3d is None or labels_3d is None or scores_3d is None:
            print_log('[Det3DRosPublishHook] make_publish_msg: bboxes_3d | labels_3d | scores_3d is None',logger='current',
            level=logging.WARNING)
            return None
        print_log(f'INFO !! detected bboxes_3d {len(bboxes_3d)}',logger='current',level=logging.INFO)


        BoundingBbox3dArray = self._BoundingBoxArray_
        BoundingBox = self._BoundingBox_
        QuaternionFromEuler = self._quaternion_from_euler_
        bbox_msg = BoundingBbox3dArray()
        # get bbox timestamp
        bbox_msg.header.stamp = self.get_bbox_time_stamp(data_sample)
        self.publish_time = bbox_msg.header.stamp # publisher time : bbox equal to pcd 
        bbox_msg.header.frame_id = self.frame_id

        bboxes_3d_tensor = tensor2ndarray(bboxes_3d.tensor)

        proc_count = 0
        for box, label, score in zip(bboxes_3d, labels_3d, scores_3d):
            center = box[0:3]
            dims = box[3:6]
            yaw = box[6]
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -yaw)

            bbox = BoundingBox()
            bbox.header = bbox_msg.header
            bbox.pose.position.x = float(center[0])
            bbox.pose.position.y = float(center[1])
            bbox.pose.position.z = float(center[2])

            bbox.pose.orientation.x = float(qx)
            bbox.pose.orientation.y = float(qy)
            bbox.pose.orientation.z = float(qz)
            bbox.pose.orientation.w = float(qw)

            bbox.dimensions.x = float(dims[0])
            bbox.dimensions.y = float(dims[1])
            bbox.dimensions.z = float(dims[2])

            bbox.label = int(label)
            bbox.value = float(score)
            bbox_msg.boxes.append(bbox)
            proc_count += 1
        print_log(f'INFO !! published {proc_count} bboxes',logger='current',level=logging.INFO)
        return bbox_msg

    def make_publish_bbox(self,
                          data_sample:Optional[Det3DDataSample] = None) -> Optional[BoundingBoxArray]:
        if data_sample is None:
            print_log('[Det3DRosPublishHook] make_publish_bbox: data_sample is None',logger='current',
            level=logging.WARNING)
            return None
        BoundingBbox3dArray = self._BoundingBoxArray_
        empty_msg = BoundingBbox3dArray()
        empty_msg.header.stamp = self.get_bbox_time_stamp(data_sample)
        empty_msg.header.frame_id = self.frame_id

        if 'pred_instances_3d' not in data_sample:
            print_log('ERROR !! pred_instances_3d not found in data_sample',logger='current',level=logging.WARNING)
            return empty_msg
        pred_instances_3d = data_sample.pred_instances_3d

        # Filter predictions by score threshold
        if hasattr(pred_instances_3d, 'scores_3d'):
            pred_instances_3d = pred_instances_3d[pred_instances_3d.scores_3d > self.pred_score_thr].to('cpu')

        bboxes_3d = getattr(pred_instances_3d,'bboxes_3d',None)# bounding box 3d
        labels_3d = getattr(pred_instances_3d,'labels_3d',None) # label 3d
        scores_3d = getattr(pred_instances_3d,'scores_3d',None) # score 3d

        # if isinstance(bboxes_3d,BaseInstance3DBoxes):
        #     bbox_msg = self.make_publish_msg(bboxes_3d, labels_3d, scores_3d, data_sample)
        bbox_msg = self.make_publish_msg(bboxes_3d, labels_3d, scores_3d, data_sample)
        return bbox_msg        

    def publish_bbox(self,bbox3d_msgs:Optional[BoundingBoxArray]= None):
        if bbox3d_msgs is None:
            print_log('[Det3DRosPublishHook] publish_bbox: bbox3d_msgs is None',logger='current',
            level=logging.WARNING)
            return
        self._publisher_bbox.publish(bbox3d_msgs)


    def publish_pcd(self, pcd:Union[np.ndarray,Tensor]):
        if pcd is None:
            print_log('ERROR: pcd is None , please check the pcd input data', logger='current', level=logging.WARNING)
            return

        if isinstance(pcd, Tensor):
            pcd = tensor2ndarray(pcd)
        pcd_msg = self.make_publish_pcd_msgs(pcd, frame_id = self.frame_id)
        if pcd_msg is not None:
            self._publisher_pcd.publish(pcd_msg)
        else:
            print_log('ERROR: pcd_msg is None , please check the pcd_msg input data', logger='current', level=logging.WARNING)
            return

    def publish_filtered_pcd(self, pcd: Union[np.ndarray, Tensor]):
        """Publish filtered pointcloud (points inside predicted boxes)."""
        if pcd is None:
            print_log('ERROR: filtered pcd is None', logger='current', level=logging.WARNING)
            return
        if isinstance(pcd, Tensor):
            pcd = tensor2ndarray(pcd)
        pcd_msg = self.make_publish_pcd_msgs(pcd, frame_id=self.frame_id)
        if pcd_msg is not None:
            self._publisher_filtered_pcd.publish(pcd_msg)
        else:
            print_log('ERROR: filtered pcd_msg is None', logger='current', level=logging.WARNING)
            return

    def make_publish_pcd_msgs(self , pcd : Optional[np.ndarray] = None,frame_id :str = None) -> Optional[PointCloud2]:
        if pcd is None:
            print_log('ERROR: pcd is None , please check the pcd input data', logger='current', level=logging.WARNING)
            return None

        point_fields = [PointField(name='x', offset=0,
                               datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4,
                               datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8,
                               datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=12,
                               datatype=PointField.FLOAT32, count=1)]  # pcd buffer 
        if self.publish_time is not None:
            header = Header(frame_id=frame_id, stamp=self.publish_time) if frame_id is not None else Header(frame_id=self.frame_id, stamp=self.publish_time)
        else:
            header = Header(frame_id=frame_id, stamp=rospy.Time.now()) if frame_id is not None else Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        
        
        points_byte = pcd[:, 0:4].tobytes()
        num_points = len(pcd)
        point_step = 16  # 4 floats * 4 bytes each = 16 bytes per point
        
        print_log(f'INFO !! point cloud msgs make successfully for publisher !! ', logger='current', level=logging.INFO)
        return PointCloud2(
                        header=header,
                        height=1,
                        width=num_points,
                        is_dense=False,
                        is_bigendian=False,
                        fields=point_fields,
                        point_step=point_step,
                        row_step=len(points_byte),
                        data=points_byte)

def run(init_args: dict, call_args: dict):
    try:
        out_pred_img_path = call_args.pop('img_out_dir')
        pred_score_thr = float(call_args.get('pred_score_thr', 0.3))  # Keep in call_args for run_inference
        pcd_topic = str(call_args.pop('pcd_topic', 'velodyne_points'))
        filtered_pcd_topic = str(call_args.pop('filtered_pcd_topic', 'velodyne_points_filtered'))
        publish_raw_pcd = bool(call_args.pop('publish_raw_pcd', False))
        publish_filtered_pcd = bool(call_args.pop('publish_filtered_pcd', True))
        # Remove any remaining input-related parameters that shouldn't be passed to run_inference
        # These should have been removed in parse_args(), but we do it here as a safety measure
        inferencer = MultiModalityDetectionInferencerNode(**init_args)
        inferencer._init_model_pipeline(
            out_pcd_img_path=out_pred_img_path,
            cam_type='CAM2',
            pred_score_thr=pred_score_thr,
            pcd_topic=pcd_topic,
            filtered_pcd_topic=filtered_pcd_topic,
            publish_raw_pcd=publish_raw_pcd,
            publish_filtered_pcd=publish_filtered_pcd)
        results_vis = inferencer.run_inference(**call_args)
        if 'visualization' not in results_vis:
            print_log('ERROR: visualization not found in results', logger='current', level=logging.WARNING)
        else:
            print_log('Inference completed successfully', logger='current', level=logging.INFO)

    except Exception as e:
        print_log(f'Error in ROS detection node run: {e}', logger='current', level=logging.ERROR)
        print_log(traceback.format_exc(), logger='current', level=logging.ERROR)
        raise
            
def main():
    """Main function to run the ROS detection node."""
    try:
        # Parse command line arguments
        init_args, call_args = parse_args()
        run(init_args, call_args)

    except KeyboardInterrupt:
        print_log('ROS detection node interrupted by user', logger='current', level=logging.INFO)
    except Exception as e:
        print_log(f'Error running ROS detection node: {e}', logger='current', level=logging.ERROR)
        print_log(traceback.format_exc(), logger='current', level=logging.ERROR)


if __name__ == '__main__':
    main()

