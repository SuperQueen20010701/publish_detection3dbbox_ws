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
        if isinstance(v,(list,tuple)):
            logger.info(f'{k}:{type(v)},length:{len(v)}')
            logger.info('\n')
            if v and len(v) < 5:
                logger.info(f'{v[:]}')
                logger.info('\n')
            else:
                logger.info(f'{v[:5]}')
                logger.info('\n')
        elif isinstance(v,dict):
            logger.info(f'{k}:{type(v)},length:{len(v)}')
            logger.info('\n')
            if v and len(v) < 5:
                logger.info(f'{list(v.keys())[:]}')
                logger.info('\n')
            else:
                logger.info(f'{list(v.keys())[:5]}')
                logger.info('\n')
        else:
            logger.info(f'{k}:{type(v)},length:{len(v)}')
            logger.info('\n')
            if v:
                logger.info(f'{v[:1]}')
                logger.info('\n')


if __name__ == '__main__':
    load_local_pkl('/kitti_data/KittiData/kitti_infos_test.pkl')
        