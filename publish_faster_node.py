import argparse
import copy
from argparse import ArgumentParser
import os
import sys
import time
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

# ros publish
# import rospy
# from sensor_msgs.msg import PointCloud2,PointField
# from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
# from tf.transformations import quaternion_from_euler
# from std_msgs.msg import Header
# from geometry_msgs.msg import Point, Quaternion, Vector3
# from visualization_msgs.msg import Marker, MarkerArray
# detection inferencer
try:
    from mmdet3d.apis import MultiModalityDet3DInferencer,init_model
    from mmdet3d.structures import (BaseInstance3DBoxes, DepthInstance3DBoxes,
                                    Det3DDataSample, LiDARInstance3DBoxes)
    from mmengine.config import Config,ConfigDict 
    from mmdet3d.utils import ConfigType
    from mmengine.visualization.utils import (check_type,tensor2ndarray)
    from mmengine import load
    from mmengine.structures import InstanceData
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
    parser.add_argument('pcd_root',help='Point cloud file')
    parser.add_argument('img_root',help='image file')
    parser.add_argument('infos',help='info pickle file')
    parser.add_argument('model',help='Config file')
    parser.add_argument('weights',help='Checkpoint file')
    parser.add_argument('img_out_dir',help='output directory of visualization images')
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
    parser.set_defaults(publish_filtered_pcd=False)  # Default: publish filtered PCD (points outside boxes)
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

    call_args['inputs'] = dict(
        points = call_args.pop('pcd_root'),
        img = call_args.pop('img_root'),
        infos= call_args.pop('infos'))
    
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

class MultiModalityDetectionInferencerNode(MultiModalityDet3DInferencer):
    def __init__(self ,
                 model:Union[ModelType,str,None]=None,
                 weights:Optional[str]=None,
                 device:Optional[str] = None,
                 scope:str = 'mmdet3d',
                 palette:str = 'none',**kwargs):

        # 处理变量参数定义
        preprocess_kwargs_ = self.preprocess_kwargs.copy()
        forward_kwargs_ = self.forward_kwargs.copy()
        visualize_kwargs_ = self.visualize_kwargs.copy()
        postprocess_kwargs_ = self.postprocess_kwargs.copy()


        super(MultiModalityDetectionInferencerNode,self).__init__(
                                                            model=model,
                                                            weights=weights,
                                                            device=device,
                                                            scope=scope,
                                                            palette=palette)
        
        self.model_ = self.model if self.model is not None else print_log(f'DEBUG: load model failed', logger='current', level=logging.INFO)
        self.cfg_ = self.cfg if self.cfg is not None else print_log(f'DEBUG: Load config failed', logger='current', level=logging.INFO)
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

    def load_param_from_cfg(self) ->None:
        if self.cfg is None:
            print_log('ERROR !! cfg is None',logger='current',level=logging.WARNING)
            return 
        # Get parameters from the original pipeline before it gets modified
        try:
            pipeline_cfg = self.cfg.test_dataloader.dataset.pipeline
            load_point_idx = self._get_transform_idx(pipeline_cfg,'LoadPointsFromFile')
            load_cfg = pipeline_cfg[load_point_idx]
            self.coord_type = load_cfg.get('coord_type', 'LIDAR')
            self.load_dim = load_cfg.get('load_dim', 4)
            self.use_dim = load_cfg.get('use_dim', 4)
            if isinstance(self.use_dim, int):
                self.use_dim = list(range(self.use_dim))
        except Exception as e:
            print_log(f'ERROR !! Failed to get parameters from pipeline: {e}',logger='current',level=logging.ERROR)
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
        # if self.ros_enabled and self.ros_topic is not None:
            # self.ros_publisher_node = Det3DRosPublishNode(
            #     topic=self.ros_topic,
            #     frame_id='velodyne',
            #     pred_score_thr=float(pred_score_thr),
            #     ros_node_name='detection_bbox_publisher',
            #     queue_size=10,
            #     latch=True,
            #     enabled_ros=self.ros_enabled,
            #     pcd_topic=str(pcd_topic),
            #     filtered_pcd_topic=str(filtered_pcd_topic)
            # )

        # if self.ros_publisher_node is not None:
        #     self.makePublishBbox = self.ros_publisher_node.make_publish_bbox
        #     self.pubBboxMsg = self.ros_publisher_node.publish_bbox  # Fix: should be publish_bbox, not make_publish_bbox
        #     self.pubPcdMsg = self.ros_publisher_node.publish_pcd
        #     self.pubFilteredPcdMsg = self.ros_publisher_node.publish_filtered_pcd
        #     self.rate = self.ros_publisher_node._rate
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
                                    score_thr: float) -> Tuple[Optional[np.ndarray],Optional[np.ndarray]]:
        """Return points OUTSIDE predicted 3D boxes (remove points inside boxes, LiDAR coord only)."""
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
            print_log('DEBUG: _filter_points_in_pred_boxes: bboxes_3d is None', 
                      logger='current', level=logging.DEBUG)
            return None,None

        # Only support LiDAR boxes here (points are in velodyne frame).
        if not isinstance(bboxes_3d, LiDARInstance3DBoxes):
            print_log(
                f'WARNING: Filtered PCD publish requested, but bboxes_3d type is {type(bboxes_3d)}. '
                'Skip filtering (expects LiDARInstance3DBoxes in velodyne frame).',
                logger='current', level=logging.WARNING)
            return None,None

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
            return pcd.copy(), None

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

    def _inputs_to_dict(self, origin_inputs: InputsType, proc_pair_count: int = 0) -> Dict:
        for single_input in origin_inputs:
            points = None
            img = None

            pcd_in = single_input['points']
            if isinstance(pcd_in,str):
                try:
                    pts_bytes = mmengine.fileio.get(pcd_in)
                    points = np.frombuffer(pts_bytes , dtype=np.float32)
                    points = points.reshape(-1,self.load_dim)
                    points = points[:, self.use_dim]
                except Exception as e:
                    print_log(f'Error loading point cloud {pcd_in}: {e}', logger='current', level=logging.ERROR)
            elif isinstance(pcd_in ,np.ndarray):
                points = pcd_in.copy()
            else:
                points = None
                raise ValueError('Unsupported input type: '
                                 f'{type(pcd_in)}')
            img_path = single_input['img']
            if isinstance(img_path, str):
                try:
                    img_bytes = mmengine.fileio.get(img_path)
                    img = mmcv.imfrombytes(img_bytes)
                    img = img[:, :, ::-1]
                except Exception as e:
                    print_log(f'Error loading image {img_path}: {e}', logger='current', level=logging.ERROR)
                # img_name = osp.basename(img_path)
            elif isinstance(img_path, np.ndarray):
                img = img_path.copy()
            else:
                raise ValueError('Unsupported input type: '
                                 f'{type(img_path)}')
            # out_file = os.path.join(self.out_pcd_img_path, 'camera_out_vis', cam_type_dir,
            #                     img_name) if self.out_pcd_img_pat != '' else None
            if points is None or img is None:
                print_log('ERROR !! points or img is None',logger='current',level=logging.WARNING)
                return None 
            data_input = dict(points=points, img=img)
            return data_input


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
            
            # Prepare output directory for saving images
            img_out_dir = self.out_pcd_img_path or '/kitti_data/fast_result_pred'
            cam_dir = camera_type
            if img_out_dir:
                os.makedirs(os.path.join(img_out_dir, 'vis_camera', cam_dir), exist_ok=True)
                os.makedirs(os.path.join(img_out_dir,'filtered_pcd'),exist_ok = True)
                os.makedirs(os.path.join(img_out_dir,'no_filter_pcd'),exist_ok=True)
            # Handle visualization and/or saving pred to image
            if current_origin_input is not None:
                should_save_vis = (self.vis_frame_count % vis_save_interval == 0)
                self.vis_frame_count += 1
                
                if not no_visualize and should_save_vis:
                    # Full visualization mode: save visualization images
                    visualize_kwargs['img_out_dir'] = img_out_dir
                    visualize_kwargs['wait_time'] = vis_wait_time
                    try:
                        visualization = self.visualize(
                            [current_origin_input], [pred], show=show, **visualize_kwargs)
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
                elif no_visualize:
                    # No visualization mode: only save pred to image (draw pred on image)
                    visualize_kwargs['img_out_dir'] = img_out_dir
                    visualize_kwargs['wait_time'] = 0.0
                    visualize_kwargs['show'] = False
                    visualize_kwargs['no_save_vis'] = False  # Enable saving
                    try:
                        # Save pred boxes on image without full visualization
                        visualization = self.visualize(
                            [current_origin_input], [pred], show=False, **visualize_kwargs)
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
                    # Get points from current_origin_input (could be path string or numpy array)
                    pcd_data = current_origin_input.get('points') if isinstance(current_origin_input, dict) else None
                    if pcd_data is None and isinstance(current_origin_input, dict):
                        pcd_data = current_origin_input['points']

                    pcd = None
                    if isinstance(pcd_data, str):
                        # Points are a file path - load and publish
                        from pathlib import Path
                        pcd_path = pcd_data
                        pcd_path_resolved = str(Path(pcd_path).resolve())
                        pcd_dir = os.path.dirname(pcd_path_resolved)
                        
                        # Check if path contains 'velodyne_reduced' and replace if needed
                        if 'velodyne_reduced' in pcd_dir.lower():
                            pcd_path = pcd_path.replace("velodyne_reduced", "velodyne")
                        
                        if os.path.isfile(pcd_path) and pcd_path.endswith('.bin'):
                            pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
                        else:
                            print_log(f'WARNING !! pcd file not found: {pcd_path}', 
                                     logger='current', level=logging.WARNING)
                    elif isinstance(pcd_data, np.ndarray):
                        # Points are already loaded as numpy array
                        if len(pcd_data.shape) == 2 and pcd_data.shape[1] >= 4:
                            pcd = pcd_data[:, :4]
                        else:
                            print_log(f'WARNING !! Invalid pcd array shape: {pcd_data.shape}', 
                                     logger='current', level=logging.WARNING)
                            pcd = None
                    else:
                        print_log(f'WARNING !! Unsupported points type: {type(pcd_data)}', 
                                 logger='current', level=logging.WARNING)

                    # Publish raw pointcloud (backward-compatible default)
                    if pcd is not None and getattr(self, 'publish_raw_pcd', False):
                        self.pubPcdMsg(pcd)
                        print_log(f'INFO !! raw pcd published (shape: {pcd.shape})',
                                  logger='current', level=logging.INFO)

                    # Publish filtered pointcloud (points OUTSIDE predicted boxes, removed points inside boxes)
                    if pcd is not None and getattr(self, 'publish_filtered_pcd', True):
                        filtered,filter_out = self._filter_points_in_pred_boxes(
                            pcd=pcd, data_sample=pred, score_thr=pred_score_thr)
                        if filtered is not None and len(filtered) > 0:
                            # Fix: publish_filtered_pcd only accepts one parameter (pcd)
                            self.pubFilteredPcdMsg(filtered)
                            # save filtered pcd to file
                            if filter_out is None or len(filter_out) == 0:
                                 no_filter_pcd_path = os.path.join(img_out_dir, 'no_filter_pcd', f'{idx:06d}.bin')
                                 filtered.astype(np.float32).tofile(no_filter_pcd_path)
                            else:
                                filtered_pcd_path = os.path.join(img_out_dir, 'filtered_pcd', f'{idx:06d}.bin') #  f'{str(idx).zfill(6)}.bin'
                                filtered.astype(np.float32).tofile(filtered_pcd_path)
                            print_log(f'INFO !! filtered pcd published (points outside boxes, shape: {filtered.shape})',
                                      logger='current', level=logging.INFO)
                        elif filtered is not None and len(filtered) == 0:
                            print_log(f'WARNING !! filtered pcd is empty (all points are inside boxes)',
                                      logger='current', level=logging.WARNING)
                        else:
                            print_log(f'WARNING !! filtered pcd is None (filtering failed)',
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

        # self._rospy = None
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

    # def _enable_publish_ros(self) -> bool:
    #     if not self.enabled_ros:
    #         return False
    #     if self._ros_node_ready:
    #         return True
        
    #     if not rospy.core.is_initialized():
    #         try:
    #             rospy.init_node(self.ros_node_name, anonymous=True, disable_signals=True)

    #         except Exception as e:
    #             print_log(f"Failed to initialize ROS node: {e}",logger='current',level=logging.ERROR)
    #             self.enabled_ros = False
    #             return False

    #     # init ros related
    #     self._rospy = rospy
    #     self._rate = rospy.Rate(self._publish_rate)
    #     self._BoundingBox_ = BoundingBox
    #     self._BoundingBoxArray_ = BoundingBoxArray
    #     self._quaternion_from_euler_ = quaternion_from_euler
    #     # bounding box publisher 
    #     self._publisher_bbox = rospy.Publisher(
    #         self.topic, BoundingBoxArray, queue_size=self.queue_size, latch=self.latch)
    #     self._publisher_pcd = rospy.Publisher(
    #         self.pcd_topic, PointCloud2 ,  queue_size=self.queue_size, latch=self.latch)
    #     self._publisher_filtered_pcd = rospy.Publisher(
    #         self.filtered_pcd_topic, PointCloud2, queue_size=self.queue_size, latch=self.latch)
    #     self._ros_node_ready = True
    #     print_log(f'ROS publisher has been initialized',logger='current',level=logging.INFO)
    #     return True

    # def get_bbox_time_stamp(self ,data_sample : Det3DDataSample) -> rospy.Time:
    #     rospy = self._rospy
    #     try:
    #         timestamp = data_sample.metainfo.get('timestamp',None)
    #         if timestamp is not None:
    #             return rospy.Time.from_sec(float(timestamp))
    #     except Exception as e:
    #         print_log(
    #             f'[Det3DRosPublishHook] get timestamp failed: {e}.',
    #             logger='current',
    #         level=logging.WARNING)
    #     return rospy.Time.now()
    
    # def make_publish_msg(self,
    #                     bboxes_3d : BaseInstance3DBoxes,
    #                     labels_3d,
    #                     scores_3d,
    #                     data_sample : Det3DDataSample = None) -> BoundingBoxArray:
    #     if bboxes_3d is None or labels_3d is None or scores_3d is None:
    #         print_log('[Det3DRosPublishHook] make_publish_msg: bboxes_3d | labels_3d | scores_3d is None',logger='current',
    #         level=logging.WARNING)
    #         return None
    #     print_log(f'INFO !! detected bboxes_3d {len(bboxes_3d)}',logger='current',level=logging.INFO)


    #     BoundingBbox3dArray = self._BoundingBoxArray_
    #     BoundingBox = self._BoundingBox_
    #     QuaternionFromEuler = self._quaternion_from_euler_
    #     bbox_msg = BoundingBbox3dArray()
    #     # get bbox timestamp
    #     bbox_msg.header.stamp = self.get_bbox_time_stamp(data_sample)
    #     self.publish_time = bbox_msg.header.stamp # publisher time : bbox equal to pcd 
    #     bbox_msg.header.frame_id = self.frame_id

    #     bboxes_3d_tensor = tensor2ndarray(bboxes_3d.tensor)

    #     proc_count = 0
    #     for box, label, score in zip(bboxes_3d, labels_3d, scores_3d):
    #         center = box[0:3]
    #         dims = box[3:6]
    #         yaw = box[6]
    #         qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -yaw)

    #         bbox = BoundingBox()
    #         bbox.header = bbox_msg.header
    #         bbox.pose.position.x = float(center[0])
    #         bbox.pose.position.y = float(center[1])
    #         bbox.pose.position.z = float(center[2])

    #         bbox.pose.orientation.x = float(qx)
    #         bbox.pose.orientation.y = float(qy)
    #         bbox.pose.orientation.z = float(qz)
    #         bbox.pose.orientation.w = float(qw)

    #         bbox.dimensions.x = float(dims[0])
    #         bbox.dimensions.y = float(dims[1])
    #         bbox.dimensions.z = float(dims[2])

    #         bbox.label = int(label)
    #         bbox.value = float(score)
    #         bbox_msg.boxes.append(bbox)
    #         proc_count += 1
    #     print_log(f'INFO !! published {proc_count} bboxes',logger='current',level=logging.INFO)
    #     return bbox_msg

    # def make_publish_bbox(self,
    #                       data_sample:Optional[Det3DDataSample] = None) -> Optional[BoundingBoxArray]:
    #     if data_sample is None:
    #         print_log('[Det3DRosPublishHook] make_publish_bbox: data_sample is None',logger='current',
    #         level=logging.WARNING)
    #         return None
    #     BoundingBbox3dArray = self._BoundingBoxArray_
    #     empty_msg = BoundingBbox3dArray()
    #     empty_msg.header.stamp = self.get_bbox_time_stamp(data_sample)
    #     empty_msg.header.frame_id = self.frame_id

    #     if 'pred_instances_3d' not in data_sample:
    #         print_log('ERROR !! pred_instances_3d not found in data_sample',logger='current',level=logging.WARNING)
    #         return empty_msg
    #     pred_instances_3d = data_sample.pred_instances_3d

    #     # Filter predictions by score threshold
    #     if hasattr(pred_instances_3d, 'scores_3d'):
    #         pred_instances_3d = pred_instances_3d[pred_instances_3d.scores_3d > self.pred_score_thr].to('cpu')

    #     bboxes_3d = getattr(pred_instances_3d,'bboxes_3d',None)# bounding box 3d
    #     labels_3d = getattr(pred_instances_3d,'labels_3d',None) # label 3d
    #     scores_3d = getattr(pred_instances_3d,'scores_3d',None) # score 3d

    #     # if isinstance(bboxes_3d,BaseInstance3DBoxes):
    #     #     bbox_msg = self.make_publish_msg(bboxes_3d, labels_3d, scores_3d, data_sample)
    #     bbox_msg = self.make_publish_msg(bboxes_3d, labels_3d, scores_3d, data_sample)
    #     return bbox_msg        

    # def publish_bbox(self,bbox3d_msgs:Optional[BoundingBoxArray]= None):
    #     if bbox3d_msgs is None:
    #         print_log('[Det3DRosPublishHook] publish_bbox: bbox3d_msgs is None',logger='current',
    #         level=logging.WARNING)
    #         return
    #     self._publisher_bbox.publish(bbox3d_msgs)


    # def publish_pcd(self, pcd:Union[np.ndarray,Tensor]):
    #     if pcd is None:
    #         print_log('ERROR: pcd is None , please check the pcd input data', logger='current', level=logging.WARNING)
    #         return

    #     if isinstance(pcd, Tensor):
    #         pcd = tensor2ndarray(pcd)
    #     pcd_msg = self.make_publish_pcd_msgs(pcd, frame_id = self.frame_id)
    #     if pcd_msg is not None:
    #         self._publisher_pcd.publish(pcd_msg)
    #     else:
    #         print_log('ERROR: pcd_msg is None , please check the pcd_msg input data', logger='current', level=logging.WARNING)
    #         return

    # def publish_filtered_pcd(self, pcd: Union[np.ndarray, Tensor]):
    #     """Publish filtered pointcloud (points inside predicted boxes)."""
    #     if pcd is None:
    #         print_log('ERROR: filtered pcd is None', logger='current', level=logging.WARNING)
    #         return
    #     if isinstance(pcd, Tensor):
    #         pcd = tensor2ndarray(pcd)
    #     pcd_msg = self.make_publish_pcd_msgs(pcd, frame_id=self.frame_id)
    #     if pcd_msg is not None:
    #         self._publisher_filtered_pcd.publish(pcd_msg)
    #     else:
    #         print_log('ERROR: filtered pcd_msg is None', logger='current', level=logging.WARNING)
    #         return

    # def make_publish_pcd_msgs(self , pcd : Optional[np.ndarray] = None,frame_id :str = None) -> Optional[PointCloud2]:
    #     if pcd is None:
    #         print_log('ERROR: pcd is None , please check the pcd input data', logger='current', level=logging.WARNING)
    #         return None

    #     point_fields = [PointField(name='x', offset=0,
    #                            datatype=PointField.FLOAT32, count=1),
    #                 PointField(name='y', offset=4,
    #                            datatype=PointField.FLOAT32, count=1),
    #                 PointField(name='z', offset=8,
    #                            datatype=PointField.FLOAT32, count=1),
    #                 PointField(name='intensity', offset=12,
    #                            datatype=PointField.FLOAT32, count=1)]  # pcd buffer 
    #     if self.publish_time is not None:
    #         header = Header(frame_id=frame_id, stamp=self.publish_time) if frame_id is not None else Header(frame_id=self.frame_id, stamp=self.publish_time)
    #     else:
    #         header = Header(frame_id=frame_id, stamp=rospy.Time.now()) if frame_id is not None else Header(frame_id=self.frame_id, stamp=rospy.Time.now())
        
        
    #     points_byte = pcd[:, 0:4].tobytes()
    #     num_points = len(pcd)
    #     point_step = 16  # 4 floats * 4 bytes each = 16 bytes per point
        
    #     print_log(f'INFO !! point cloud msgs make successfully for publisher !! ', logger='current', level=logging.INFO)
    #     return PointCloud2(
    #                     header=header,
    #                     height=1,
    #                     width=num_points,
    #                     is_dense=False,
    #                     is_bigendian=False,
    #                     fields=point_fields,
    #                     point_step=point_step,
    #                     row_step=len(points_byte),
    #                     data=points_byte)

def run(init_args: dict, call_args: dict):
    try:
        out_pred_img_path = call_args.pop('img_out_dir')
        pred_score_thr = float(call_args.get('pred_score_thr', 0.3))  # Keep in call_args for run_inference
        pcd_topic = str(call_args.pop('pcd_topic', 'velodyne_points'))
        filtered_pcd_topic = str(call_args.pop('filtered_pcd_topic', 'velodyne_points_filtered'))
        publish_raw_pcd = bool(call_args.pop('publish_raw_pcd', False))
        publish_filtered_pcd = bool(call_args.pop('publish_filtered_pcd', True))
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

