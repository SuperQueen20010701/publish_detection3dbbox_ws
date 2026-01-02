
from __future__ import annotations
import mmcv
import numpy as np
import math
from typing import Optional, Sequence, Tuple

from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.logging import print_log
import logging
from mmengine.runner import Runner
from mmengine.visualization.utils import (check_type, tensor2ndarray)

from mmdet3d.registry import HOOKS
from mmdet3d.structures import (BaseInstance3DBoxes, Box3DMode,
                                CameraInstance3DBoxes, Coord3DMode,
                                DepthInstance3DBoxes, DepthPoints,
                                Det3DDataSample, LiDARInstance3DBoxes,
                                PointData, points_cam2img)
from mmdet3d.visualization import to_depth_mode
@HOOKS.register_module()
class Det3DRosPublishHook(Hook):
    """Publish predicted 3D boxes as `jsk_recognition_msgs/BoundingBoxArray` during test."""

    def __init__(
        self,
        topic: str = '/box',
        frame_id: str = 'velodyne',
        pred_score_thr: float = 0.3,
        ros_node_name: str = '3d_detection_bbox_publisher',
        queue_size: int = 10,
        latch: bool = True,
        enabled_ros: bool = True,
        backend_args:Optional[dict] = None
    ) -> None:
        self.topic = topic
        self.frame_id = frame_id
        self.pred_score_thr = float(pred_score_thr)
        self.ros_node_name = ros_node_name
        self.queue_size = int(queue_size)
        self.latch = bool(latch)
        self.enabled_ros = bool(enabled_ros)
        self._ros_node_ready = False
        self._rospy = None
        self._BoundingBox_func = None
        self._BoundingBoxArray_func = None
        self._quaternion_from_euler_func = None
        self._publisher_bbox = None
        self.backend_args = backend_args

    def proc_is_rank_0(self, runner: Runner) -> bool:
        return int(getattr(runner, 'rank', 0)) == 0
    def enable_ros_publish(self, runner: Optional[Runner] = None) -> bool:
        if not self.enabled_ros:
            return False
        if self._ros_node_ready:
            return True

        try:
            import rospy  # type: ignore
            from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray  # type: ignore
            from tf.transformations import quaternion_from_euler  # type: ignore
        except Exception as e:  # pragma: no cover
            if runner is not None:
                print_log(
                    f'[Det3DRosPublishHook] ROS import failed: {e}.',
                    logger='current',
            level=logging.WARNING)
            self.enabled_ros = False
            return False
        if not rospy.core.is_initialized():
            try:
                rospy.init_node(self.ros_node_name, anonymous=True, disable_signals=True)
            except Exception as e: 
                if runner is not None:
                    print_log(
                        f'[Det3DRosPublishHook] rospy.init_node failed: {e}.',
                        logger='current',level=logging.WARNING)
                self.enabled_ros = False
                return False

        self._rospy = rospy
        self._BoundingBox_func = BoundingBox
        self._BoundingBoxArray_func = BoundingBoxArray
        self._quaternion_from_euler_func = quaternion_from_euler
        # bounding box publisher 
        self._publisher_bbox = rospy.Publisher(
            self.topic, BoundingBoxArray, queue_size=self.queue_size, latch=self.latch)
        self._ros_node_ready = True
        if runner is not None:
            print_log(
                f'[Det3DRosPublishHook] ROS publisher ready: {self.topic} ({self.frame_id})',
                logger='current',
            level=logging.WARNING)
        return True
    # 获取bounding box 的时间戳
    def get_bbox_time_stampe(self ,data_sample : Det3DDataSample) -> rospy.Time:
        rospy_cir = self.rospy
        try:
            timestamp = data_sample.metainfo.get('timestamp',None)
            if timestamp is not None:
                return rospy_cir.Time.from_sec(float(timestamp))
        except Exception as e:
            print_log(
                f'[Det3DRosPublishHook] get timestamp failed: {e}.',
                logger='current',
            level=logging.WARNING)
        return rospy_cir.Time.now()

    def make_publish_msg(self,
                        bboxes_3d : BaseInstance3DBoxes,
                        labels_3d: Tensor,
                        scores_3d: Optional[Tensor] = None,
                        msg_bbox: BoundingBoxArray = None) :
        if msg_bbox is None or labels_3d is None or scores_3d is None:
            print_log('ERROR !! msg_bbox | labels_3d | scores_3d is None',logger='current',
            level=logging.WARNING)
            return 
        # publish the msg of the bboxes_3d
        check_type('bboxes', bboxes_3d, BaseInstance3DBoxes)
        if not isinstance(bboxes_3d, DepthInstance3DBoxes):
            bboxes_3d = bboxes_3d.convert_to(Box3DMode.DEPTH)
        bboxes_3d = tensor2ndarray(bboxes_3d.tensor)

        BoundingBox= self._BoundingBox_func
        quaternion_from_euler = self._quaternion_from_euler_func
        
    
        for i in range(len(bboxes_3d)):
            center = bboxes_3d[i, 0:3]
            dims = bboxes_3d[i, 3:6]
            yaw = np.zeros(3)
            yaw = bboxes_3d[i, 6]
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, -yaw)
            bbox = BoundingBox()
            bbox.header = msg_bbox.header
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

            bbox.label = int(labels_3d[i])
            bbox.value = float(scores_3d[i])
            msg_bbox.boxes.append(bbox)

    def make_publish_bbox(self ,
                        data_input:dict,
                        data_sample :Optional[Det3DDataSample] = None) -> Optional[BoundingBoxArray]:
        if data_sample is None:
            print_log('[Det3DRosPublishHook] make_publish_bbox: data_sample is None',logger='current',
            level=logging.WARNING)
            return None
        #function declaration
        
        BoundingBoxArray= self._BoundingBoxArray_func
        quaternion_from_euler = self._quaternion_from_euler_func
        # for the publisher of the message 
        msg = BoundingBoxArray()
        msg.header.stamp = self.get_bbox_time_stampe(data_sample)
        msg.header.frame_id = self.frame_id
        
        if 'pred_instances_3d' not in data_sample:
            return msg
        pred_instances_3d = data_sample.pred_instances_3d
        pred_instances_3d = [pred_instances_3d.scores_3d >self.pred_score_thr].to('cpu')

        # rot_mat, trans_vec = self.proc_transform_bbox(data_input, pred_instances_3d, data_sample.metainfo)
        if not hasattr(pred_instances_3d, 'bboxes_3d'):
            print_log('ERROR !! No bboxes_3d found in the pred_instances_3d',logger='current',
            level=logging.WARNING)
            return msg

        bboxes_3d = getattr(pred_instances_3d,'bboxes_3d',None)# bounding box 3d
        labels_3d = getattr(pred_instances_3d,'labels_3d',None) # label 3d
        scores_3d = getattr(pred_instances_3d,'scores_3d',None) # score 3d
        if 'points' not in data_input:
            print_log('[Det3DRosPublishHook] Points data not found', logger='current', level=logging.WARNING)
            return msg
        points = data_input['points']
        check_type('points',points, np.ndarray)
        points = tensor2ndarray(points)

        if not isinstance(bboxes_3d,DepthInstance3DBoxes):
            points, bboxes_3d_depth = to_depth_mode(points, bboxes_3d)
        else:
            bboxes_3d_depth = bboxes_3d.clone()

        self.make_publish_msg(bboxes_3d_depth, labels_3d, scores_3d, msg)

        return msg

 

    # def proc_transform_bbox(self,
    #                         data_input: dict,
    #                         instances: InstanceData,
    #                         input_meta: dict) -> Tuple[Union[Tensor, np.ndarray, float], Union[Tensor, np.ndarray]]:
    #     if not len(instances) > 0:
    #         print_log('ERROR !! No instances found in the data_input',logger='current',
    #         level=logging.WARNING)
    #         return None

    #     rot_mat = np.eye(3)
    #     trans_vec = np.zeros(3)
    #     if 'axis_align_matrix' in input_meta:
    #         rot_mat = input_meta['axis_align_matrix'][:3,:3]
    #         trans_vec = input_meta['axis_align_matrix'][:3,-1]
    #     else:
    #         print_log('ERROR !! No axis_align_matrix found in the input_meta',logger='current',
    #         level=logging.WARNING)
    #     return rot_mat, trans_vec

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[Det3DDataSample]) -> None:
        if not self.proc_is_rank_0(runner):
            return
        if not self.enable_ros_publish(runner):
            return

        for data_sample in outputs:
            try:
                data_input = dict()
                assert 'img_path' in data_sample and 'lidar_path' in data_sample, \
                "'data_sample' must contain 'img_path' or 'lidar_path'"
                img_path = data_sample.img_path
                if isinstance(img_path , list):
                    img_list = []
                    # add img to data input
                    for single_path_ in img_path:
                        img_bytes = get(
                            single_img_path, backend_args=self.backend_args)
                        single_img = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        img_list.append(single_img)
                    # add lidar point in data_input 
                lidar_path = data_sample.lidar_path
                num_pts_feats = data_sample.num_pts_feats
                pts_type = get(lidar_path,backend_args = self.backend_args)
                points = np.frombuffer(pts_type,dtype = np.float32)
                points = points.reshape(-1,num_pts_feats)
                data_input['points'] = points
                bbox_3d_publish = self.make_publish_bbox(data_input, data_sample)
                if bbox_3d_publish is not None:
                    self._publisher_bbox.publish(bbox_3d_publish)
                else:
                    print_log('ERROR !! bbox_3d_publish is None',logger='current',
                    level=logging.WARNING)
                    return
            except Exception:
                if runner is not None :
                    print_log(
                        '[Det3DRosPublishHook] publish() failed; disabling ROS publishing.',
                        logger='current',
            level=logging.WARNING)
                self.enabled_ros = False
                return


