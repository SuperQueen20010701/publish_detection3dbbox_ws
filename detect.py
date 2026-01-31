import sys
import mmengine
from mmengine.fileio import get_local_path
import os
import numpy as np
from typing import List, Optional,Tuple,Dict,Any
from mmengine.logging import MMLogger

def load_local_pkl(pkl_path: str,backend_args:Optional[dict] =None) ->None:
    with get_local_path(pkl_path,backend_args=backend_args) as local_path:
        db_infos = mmengine.load(open(local_path,'rb'),file_format='pkl')

    logger: MMLogger = MMLogger.get_current_instance()
    logger.info(f'load {len(db_infos)} database infos in DataBaseSampler')
    for k,v in db_infos.items():
        if k == 'data_list':
            if len(v[0].keys()) == len(v[1].keys()):
                item = v[0]
                for key,value in item.items():
                    logger.info(f'{key}:{type(value)}')
                    if key == 'images':
                        if isinstance(value,dict):
                            if 'CAM2' in value.keys() and isinstance(value["CAM2"],dict):
                                print(f'CAM2: {value["CAM2"].keys()}')
                    if key == 'lidar_points':
                        if isinstance(value,dict):
                            print(list(value.keys()))
                    if key == 'instances':
                        print(f'length of instances: {len(value)}')
            logger.info('\n')
if __name__ == '__main__':
    load_local_pkl('/src/mmdetection3d/publish_detection3dbbox_ws/data/kitti/kitti_infos_test.pkl')
        