"""
产生yolo格式的注释文件
例如 'MOT17/labels_with_ids/train/MOT17-02-SDP/img1/000010.txt'
"""

import os
import os.path as osp
import argparse

DATA_ROOT = '/data/wujiapeng/datasets/MOT17'

certain_seqs = ['MOT17-02-SDP']

image_wh_dict = {
    'MOT17-02-SDP': (1920, 1080),
    'MOT17-04-SDP': (1920, 1080),
    'MOT17-05-SDP': (640, 480),
    'MOT17-09-SDP': (1920, 1080),
    'MOT17-10-SDP': (1920, 1080),
    'MOT17-11-SDP': (1920, 1080),
    'MOT17-13-SDP': (1920, 1080),
    'MOT17-01-SDP': (1920, 1080),
    'MOT17-03-SDP': (1920, 1080),
    'MOT17-06-SDP': (640, 480),
    'MOT17-07-SDP': (1920, 1080),
    'MOT17-08-SDP': (1920, 1080),
    'MOT17-12-SDP': (1920, 1080),
    'MOT17-14-SDP': (1920, 1080),
}  # seq->(w,h) 字典 用于归一化


def generate_labels(split='train', if_certain_seqs=False, if_write_txt=True):
    """
    split: str, 'train', 'val' or 'test'
    if_certain_seqs: bool, use for debug. 
    """

    if not if_certain_seqs:
        seq_list = os.listdir(osp.join(DATA_ROOT, split))  # 序列列表
    else:
        seq_list = certain_seqs

    # 对于MOT17数据集 只需要取一个检测器的 例如SDP

    seq_list = [seq for seq in seq_list if seq.endswith('SDP')]

    # 每张图片分配一个txt
    # 要从sequence的txt里分出来

    current_id = 0  # 表示写入txt的当前id 一个目标在所有视频序列里是唯一的
    last_id = -1  # 表示写入的上一个id

    for seq in seq_list:
        seq_dir = osp.join(DATA_ROOT, split, seq, 'gt', 'gt.txt')
        with open(seq_dir, 'r') as f:
            lines = f.readlines()

            for row in lines:
                current_line = row.split(',')  # 读取gt的当前行
                # 需要写进新文件行的文字
                # 例如 0 1 781 262 200 631  (0 id x_c y_c w h)
                frame = current_line[0]  # 第几帧
                id_in_frame = int(current_line[1])  # 当前帧当前目标的id
                if not id_in_frame == last_id:  # 如果到了下一个id
                    current_id += 1  # 写入文件的id加一
                    last_id = id_in_frame  # 更新last id
                # 写到对应图片的txt

                # to_file = osp.join(DATA_ROOT, 'labels_with_ids', split, seq, 'img1', frame.zfill(6) + '.txt')
                to_file = osp.join(DATA_ROOT, 'labels_with_ids', split, seq, 'img1')
                if not osp.exists(to_file):
                    os.makedirs(to_file)

                to_file = osp.join(to_file, frame.zfill(6) + '.txt')
                with open(to_file, 'a') as f_to:
                    id = current_line[1]  # 当前id
                    x0, y0 = int(current_line[2]), int(current_line[3])  # 左上角 x y
                    w, h = int(current_line[4]), int(current_line[5])  # 宽 高

                    x_c, y_c = x0 + w // 2, y0 + h // 2  # 中心点 x y

                    image_w, image_h = image_wh_dict[seq][0], image_wh_dict[seq][1]  # 图像高宽
                    # 归一化
                    w, h = w / image_w, h / image_h
                    x_c, y_c = x_c / image_w, y_c / image_h

                    write_line = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                        current_id, x_c, y_c, w, h)

                    f_to.write(write_line)
                f_to.close()
        f.close()

    print(f"writing txt{if_write_txt}")
    if if_write_txt:
        with open('./mot.txt', 'a') as f:
            for seq in seq_list:
                img_dir = osp.join(DATA_ROOT, split, seq)  # 该序列下所有图片路径
                imgs = sorted(os.listdir(img_dir))  # 所有图片

                for img in imgs:  # 遍历该序列下的图片
                    f.write('MOT17/' + 'images/' + split + '/' + seq + '/' \
                             + 'img1/' + img + '\n')
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='train or test')
    parser.add_argument('--if_certain_seqs', type=bool, default=False, help='for debug')
    parser.add_argument('--if_write_txt', type=bool, default=True, help='write mot17.train or mot17.val')

    opt = parser.parse_args()

    generate_labels(opt.split, opt.if_certain_seqs, opt.if_write_txt)

    print('Done!')
