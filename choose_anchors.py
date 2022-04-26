"""
根据k-means聚类算法选择先验锚框
"""

import os
import os.path as osp
import argparse

import numpy as np 

from kmeans import kmeans, avg_iou

DATA_ROOT = '/data/wujiapeng/datasets/VisDrone2019/VisDrone2019'

NUM_ANCHORS = 12  # 锚框个数

ignored_seqs = ['uav0000013_00000_v', 'uav0000013_01073_v', 'uav0000013_01392_v',
                'uav0000020_00406_v', 'uav0000071_03240_v', 'uav0000072_04488_v',
                'uav0000072_05448_v', 'uav0000072_06432_v', 'uav0000079_00480_v',
                'uav0000084_00000_v', 'uav0000099_02109_v', 'uav0000086_00000_v',
                'uav0000073_00600_v', 'uav0000073_04464_v', 'uav0000088_00290_v']

certain_seqs = ['uav0000124_00944_v','uav0000126_00001_v','uav0000138_00000_v','uav0000145_00000_v','uav0000150_02310_v','uav0000222_03150_v','uav0000239_12336_v','uav0000243_00001_v',
'uav0000248_00001_v','uav0000263_03289_v','uav0000266_03598_v','uav0000273_00001_v','uav0000279_00001_v','uav0000281_00460_v','uav0000289_00001_v','uav0000289_06922_v','uav0000307_00000_v',
'uav0000308_00000_v','uav0000308_01380_v','uav0000326_01035_v','uav0000329_04715_v','uav0000361_02323_v','uav0000366_00001_v']

# 训练集图像高宽 用于归一化
# image_wh_dict = {'uav0000143_02250_v': (2688, 1512), 'uav0000264_02760_v': (1904, 1071), 'uav0000124_00944_v': (2688, 1512), 'uav0000326_01035_v': (1904, 1071), 'uav0000218_00001_v': (2688, 1512), 'uav0000361_02323_v': (1904, 1071), 'uav0000222_03150_v': (2688, 1512), 'uav0000239_03720_v': (1344, 756), 'uav0000363_00001_v': (1344, 756), 'uav0000076_00720_v': (1904, 1071), 'uav0000266_03598_v': (2688, 1512), 'uav0000244_01440_v': (1920, 1080), 'uav0000145_00000_v': (2688, 1512), 'uav0000270_00001_v': (1904, 1071), 'uav0000295_02300_v': (2720, 1530), 'uav0000357_00920_v': (1904, 1071), 'uav0000239_12336_v': (1344, 756), 'uav0000307_00000_v': (1904, 1071), 'uav0000126_00001_v': (2688, 1512), 'uav0000329_04715_v': (1904, 1071), 'uav0000323_01173_v': (1904, 1071), 'uav0000140_01590_v': (2688, 1512), 'uav0000309_00000_v': (1904, 1071), 'uav0000243_00001_v': (1344, 756), 'uav0000138_00000_v': (2688, 1512), 'uav0000308_01380_v': (1904, 1071), 'uav0000248_00001_v': (1344, 756), 'uav0000360_00001_v': (1904, 1071), 'uav0000278_00001_v': (1360, 765), 'uav0000342_04692_v': (1904, 1071), 'uav0000289_06922_v': (1904, 1071), 'uav0000289_00001_v': (1904, 1071), 'uav0000315_00000_v': (1904, 1071), 'uav0000316_01288_v': (2720, 1530), 'uav0000279_00001_v': (1904, 1071), 'uav0000281_00460_v': (1904, 1071), 'uav0000266_04830_v': (2688, 1512), 'uav0000352_05980_v': (1904, 1071), 'uav0000300_00000_v': (1904, 1071), 'uav0000263_03289_v': (1904, 1071), 'uav0000366_00001_v': (1904, 1071), 'uav0000150_02310_v': (2688, 1512), 'uav0000288_00001_v': (1904, 1071), 'uav0000273_00001_v': (1904, 1071), 'uav0000308_00000_v': (1904, 1071)}
image_wh_dict = {'uav0000120_04775_v': (1360, 765), 'uav0000370_00001_v': (2720, 1530), 'uav0000249_00001_v': (1920, 1080), 'uav0000297_02761_v': (1904, 1071), 'uav0000188_00000_v': (960, 540), 'uav0000249_02688_v': (1344, 756), 'uav0000355_00001_v': (1360, 765), 'uav0000119_02301_v': (1360, 765), 'uav0000306_00230_v': (680, 382), 'uav0000161_00000_v': (960, 540), 'uav0000297_00000_v': (1904, 1071), 'uav0000009_03358_v': (1360, 765), 'uav0000077_00720_v': (1360, 765), 'uav0000201_00000_v': (1344, 756)}


def read_bbox(split='VisDrone2019-MOT-train', if_certain_seq=False, if_norm=False):
    """
    读取真值锚框
    split: 训练集或测试集 不需改动 只聚类训练集
    if_certain_seq: 是否选取特定序列
    return: np.ndarray, shape(num_bbox, 2) 真实大小
    """

    if if_certain_seq:
        seq_list = certain_seqs
    else:
        seq_list = os.listdir(osp.join(DATA_ROOT, split, 'sequences'))  # 所有序列名称
    
    seq_list = [seq for seq in seq_list if seq not in ignored_seqs]  # 筛选

    bbox_wh = []  # 存储边界框长宽

    for seq in seq_list:
        anno_file = osp.join(DATA_ROOT, split, 'annotations', seq + '.txt')  # 真值文件路径
        with open(anno_file, 'r') as f:
            
            lines = f.readlines()

            for row in lines:
                current_line = row.split(',')  # 读取gt的当前行

                if current_line[6] == '1' and current_line[7] in ['4', '5', '6', '9']:
                    if if_norm:
                        bbox_wh.append([int(current_line[4]), int(current_line[5])])  # 注意 没有归一化
                    else:
                        orig_w, orig_h = image_wh_dict[seq][0], image_wh_dict[seq][1]
                        bbox_wh.append([int(current_line[4]) / orig_w, int(current_line[5]) / orig_h])

        f.close()

    
    return np.array(bbox_wh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='VisDrone2019-MOT-train', help='train or test')
    parser.add_argument('--if_certain_seqs', type=bool, default=False, help='for debug')
    parser.add_argument('--if_norm', type=bool, default=False, help='if normalization')
    opt = parser.parse_args()
    
    bbox_wh = read_bbox(opt.split, opt.if_certain_seqs)

    print(bbox_wh.shape)

    out = kmeans(bbox_wh, NUM_ANCHORS)
    print("Accuracy: {:.2f}%".format(avg_iou(bbox_wh, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
