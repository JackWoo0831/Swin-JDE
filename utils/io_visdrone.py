import os
from typing import Dict
import numpy as np

from utils.log import logger


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)
    logger.info('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


"""
VisDrone数据集只跟踪四类: 4, 5, 6, 9
cars, vans, trucks, bus
"""


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {4, 5, 6, 9}
    ignore_labels = {1, 2, 3, 7, 8, 10, 11}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    
                    label = int(float(linelist[7]))
                    mark = int(float(linelist[6]))
                    if mark == 0 or label not in valid_labels:
                        continue
                    score = 1

                elif is_ignore:
                    
                    label = int(float(linelist[7]))
                    vis_ratio = 1 - float(linelist[8]) / 3
                    if label not in ignore_labels and vis_ratio >= 0:
                        continue
                    
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores