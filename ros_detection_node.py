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

# ros publish
import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Vector3
from visualization_msgs.msg import Marker, MarkerArray
# detection inferencer
try:
    from mmdet3d.apis import MultiModalityDet3DInferencer,init_model
    from mmdet3d.structures import (BaseInstance3DBoxes,DepthInstance3DBoxes,Det3DDataSample)
    from mmengine.config import Config,ConfigDict 
    from mmdet3d.utils import ConfigType

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
# 参数解析
def parse_args():
    parser = ArgumentParser(description='MVXNet ROS Detection Node publisher')
    parser.add_argument('pcd_root',help='Point cloud file')
    parser.add_argument('img_root',help='image file')
    parser.add_argument('infos',help='info pickle file')
    parser.add_argument('model',help='Config file')
    parser.add_argument('weights',help='Checkpoint file')
    parser.add_argument('--device',default='cuda:0',help='Device used for inference')
    parser.add_argument('--cam-type',type=str,default='CAM_FRONT',help='choose camera type to inference')
    parser.add_argument('--pred-score-thr',type=float,default=0.3,help='bbox score threshold')
    parser.add_argument('--out-dir',type=str,default='/kitti_data/result',help='dir to save results')

    call_args = vars(parser.parse_args())

    call_args['inputs'] = dict(
        points = call_args.pop('pcd_root'),
        img = call_args.pop('img_root'),
        infos= call_args.pop('infos'))
    
    # 检查cuda 是否可用
    def _check_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    init_args = {}
    init_keys = ['model','weights','device']
    for key in init_keys:
        if key == 'device':
            if not _check_cuda():
                init_args[key] = 'cpu'
            else:
                init_args[key] =  call_args.pop(key)
        else:
            init_args[key] = call_args.pop(key)

    return init_args,call_args

class MultiModalityDetectionInferencerNode(MultiModalityDet3DInferencer):
    def __init__(self ,
                 model:Union[ModelType,str,None]=None,
                 weights:Optional[str]=None,
                 device:Optional[str] = None,
                 scope:str = 'mmdet3d',
                 palette:str = 'none'):

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
        self.ros_topic = '/detection/bboxes'
        self.rate = None

        # out path 
        self.out_pcd_img_path = None



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

    def _init_model_pipeline(self , out_pcd_img_path:Optional[str] = None, cam_type:Optional[str] = None) ->None:
        # print_log(f'DEBUG: _init_model_pipeline called with model={self.model}, cfg={self.cfg}', logger='current', level=logging.INFO)
        # init model config
        if self.model is not  None and self.cfg is None:
            self.cfg = self.load_cfg(self.model)
        else:
            print_log(f'DEBUG: Config and Model load successfully ', logger='current', level=logging.INFO)

        self.load_param_from_cfg()
        # init ros publisher node 
        if self.ros_enabled and self.ros_topic is not None:
            self.ros_publisher_node = Det3DRosPublishNode(
                topic=self.ros_topic,
                frame_id='velodyne',
                pred_score_thr=0.3,
                ros_node_name='detection_bbox_publisher',
                queue_size=10,
                latch=True,
                enabled_ros=self.ros_enabled
            )

        if self.ros_publisher_node is not None:
            self.makePublishBbox = self.ros_publisher_node.make_publish_bbox
            self.pubBboxMsg = self.ros_publisher_node.publish_bbox       
            self.rate = self.ros_publisher_node._rate
        if out_pcd_img_path is not None:
            self.out_pcd_img_path = out_pcd_img_path
        else:
            self.out_pcd_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'out_pcd_img')
            os.makedirs(self.out_pcd_img_path,exist_ok=True)
        
        if cam_type is not None:
            self.cam_type = cam_type
        else:
            self.cam_type = 'CAM2'

    def _inputs_to_dict(self, origin_inputs: InputsType, proc_pair_count: int = 0) -> Dict:

        for single_input in origin_inputs:
            # Always initialize locals. If loading fails inside try/except,
            # we still want deterministic behavior instead of UnboundLocalError.
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
                # pc_name = os.path.basename(pcd_in).split('.bin')[0]
                # pc_name = f'{pc_name}.png'
            elif isinstance(pcd_in ,np.ndarray):
                points = pcd_in.copy()
            #     if proc_pair_count !=0 : 
            #         pc_num = str(proc_pair_count).zfill(8)
            #     else:
            #         pc_num = str(self.num_of_proc_count).zfill(8) if self.num_of_proc_count is not None else None
            #     if pc_num is not None:
            #         pc_name = f'{pc_num}.png'
            #     else:
            #        print_log('ERROR !! pc invalid to dict !!',logger='current',level=logging.WARNING)
            else:
                points = None
                raise ValueError('Unsupported input type: '
                                 f'{type(pcd_in)}')

            # if self.out_pcd_img_path is not None:
            #     o3d_save_path = os.path.join(self.out_pcd_img_path,'lidar_out_vis',pc_name)
            #     mmengine.mkdir_or_exist(os.path.dirname(o3d_save_path))
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
                # if proc_pair_count !=0 :
                #     img_num = str(proc_pair_count).zfill(8)
                # else:
                #     img_num = str(self.num_of_proc_count).zfill(8) if self.num_of_proc_count is not None else None
                # if img_num is not None:
                #     img_name = f'{img_num}.jpg'
                # else:
                #    print_log('ERROR !! pc invalid to dict !!',logger='current',level=logging.WARNING)
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
        # 处理传入解析参数
        (preprocess_kwargs,
        forward_kwargs,
        visualize_kwargs,
        postprocess_kwargs) = self._dispatch_kwargs(**kwargs)
        # 相机处理类型
        camera_type = preprocess_kwargs.pop('cam_type','CAM2')
        # origin input 

        origin_inputs = self._inputs_to_list(inputs,cam_type=camera_type)
        if not isinstance(origin_inputs, list):
            origin_inputs = list(origin_inputs)
        print_log(f'INFO !! inputs {inputs}', logger='current', level=logging.INFO)
        detection_inputs = self.preprocess(origin_inputs,batch_size=batch_size,**preprocess_kwargs)
        results_dict = {'predictions': [], 'visualization': []}
        preds = []
        for idx, single_data in enumerate(detection_inputs):
            if single_data is None:
                print_log(f'ERROR !! failed to get single input data',logger='current',level=logging.INFO)
                continue

            self.num_of_proc_count = idx + 1

            # Forward pass for current single data
            pred = self.forward(single_data, **forward_kwargs)
            preds.extend(pred)
            # Get corresponding origin input
            current_origin_input = None
            if isinstance(origin_inputs, list) and idx < len(origin_inputs):
                current_origin_input = origin_inputs[idx]
            # Process current prediction if we have corresponding origin input
            if current_origin_input is not None:
                if not isinstance(current_origin_input, dict):
                    print_log(
                        f'ERROR !! current_origin_input type is {type(current_origin_input)}, '
                        'expected dict with keys like points/img. Skip this frame.',
                        logger='current', level=logging.ERROR)
                    continue
                # Visualize current prediction
                visualization = self.visualize(origin_inputs, preds, **visualize_kwargs)
                # Convert input to dict format for ROS publishing
                current_input_dict = self._inputs_to_dict([current_origin_input], proc_pair_count=self.num_of_proc_count)

                # Create ROS bbox message
                pub_bboxes = self.makePublishBbox(current_input_dict, pred)

                # Publish bboxes if available
                if pub_bboxes is not None:
                    self.pubBboxMsg(pub_bboxes)
                    print_log(f'publish bboxes successfully for frame {self.num_of_proc_count}',logger='current',level=logging.INFO)

                # Postprocess results
                results = self.postprocess(preds, visualization,
                                         return_datasample=return_datasample,
                                         **postprocess_kwargs)

                predictions = results['predictions']
                if hasattr(predictions, '__iter__') and not isinstance(predictions, (list, tuple)):
                    predictions = list(predictions)
                results_dict['predictions'].extend(predictions)
                # print_log(f"detected {len(results_dict['predictions'])} objects in {self.num_of_proc_count} frames",logger='current',level=logging.INFO)
                if results['visualization'] is not None:
                    results_dict['visualization'].extend(results['visualization'])
                time.sleep(0.1)
            else:
                print_log(f'current_origin_input is None for frame {self.num_of_proc_count}',logger='current',level=logging.WARNING)
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
                publish_rate: float = 10.0):
        self.topic = topic
        self.frame_id = frame_id
        self.pred_score_thr = float(pred_score_thr)

        self.enabled_ros = bool(enabled_ros)
        self._ros_node_ready = False

        self._rospy = None
        self.ros_node_name = ros_node_name
        self.queue_size = int(queue_size)
        self.latch = bool(latch)

        self._BoundingBox_ = None
        self._BoundingBoxArray_ = None
        self._quaternion_from_euler_ = None
        self._publisher_bbox = None
        self._publish_rate = publish_rate
        self._rate = None
        # initialize ros publisher node 
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
        self._ros_node_ready = True
        print_log(f'ROS publisher has been initialized',logger='current',level=logging.INFO)
        return True

    def get_bbox_time_stamp(self ,data_sample : Det3DDataSample) -> rospy.Time:
        rospy = self.rospy
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
        BoundingBbox3dArray = self._BoundingBoxArray_
        QuaternionFromEuler = self._quaternion_from_euler_
        bbox_msg = BoundingBbox3dArray()
        # get bbox timestamp
        bbox_msg.header.stamp = self.get_bbox_time_stamp(data_sample)
        bbox_msg.header.frame_id = self.frame_id

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

        return bbox_msg

    def make_publish_bbox(self,
                          data_input:dict,
                          data_sample:Optional[Det3DDataSample] = None) -> Optional[BoundingBoxArray]:
        if data_sample is None:
            print_log('[Det3DRosPublishHook] make_publish_bbox: data_sample is None',logger='current',
            level=logging.WARNING)
            return None

        if 'pred_instances_3d' not in data_sample:
            return None
        pred_instances_3d = data_sample.pred_instances_3d

        # Filter predictions by score threshold
        if hasattr(pred_instances_3d, 'scores_3d'):
            score_mask = pred_instances_3d.scores_3d > self.pred_score_thr
            if score_mask.sum() == 0:
                return None
            pred_instances_3d = pred_instances_3d[score_mask]

        if not hasattr(pred_instances_3d, 'bboxes_3d'):
            print_log('ERROR !! No bboxes_3d found in the pred_instances_3d',logger='current',
            level=logging.WARNING)
            return None

        bboxes_3d = getattr(pred_instances_3d,'bboxes_3d',None)# bounding box 3d
        labels_3d = getattr(pred_instances_3d,'labels_3d',None) # label 3d
        scores_3d = getattr(pred_instances_3d,'scores_3d',None) # score 3d

        if 'points' not in data_input:
            print_log('[Det3DRosPublishHook] Points data not found', logger='current', level=logging.WARNING)
            return None
        points = data_input['points']
        check_type('points',points, np.ndarray)
        points = tensor2ndarray(points)

        if not isinstance(bboxes_3d,DepthInstance3DBoxes):
            points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
        else:
            bboxes_3d_depth = bboxes_3d.clone()

        bbox_msg = self.make_publish_msg(bboxes_3d_depth, labels_3d, scores_3d, data_sample)

        return bbox_msg        

    def publish_bbox(self,bbox3d_msgs:Optional[BoundingBoxArray]= None):
        if bbox3d_msgs is None:
            print_log('[Det3DRosPublishHook] publish_bbox: bbox3d_msgs is None',logger='current',
            level=logging.WARNING)
            return
        self._publisher_bbox.publish(bbox3d_msgs)
        print_log(f'publish bboxes successfully',logger='current',level=logging.INFO)

def run(init_args: dict, call_args: dict):
    try:
        # Initialize inferencer
        inferencer = MultiModalityDetectionInferencerNode(**init_args)
        inferencer._init_model_pipeline(out_pcd_img_path=None, cam_type='CAM2')
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

