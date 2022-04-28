# Swin-JDE
This repo is **JDE(Joint Detection and Embedding) with Swin-T backbone** in VisDrone2019-MOT dataset, The code is built on [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) and [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).

![gif](https://github.com/JackWoo0831/Swin-JDE/blob/master/imgs/jde.gif)


The structure of this model is as follow:  

![structure](https://github.com/JackWoo0831/Swin-JDE/blob/master/imgs/jde.png)

Result on VisDrone2019-MOT test:
(*Used ByteTrack, high thresh=0.6, low thresh=0.2*)

||IDF1|Recall|Precision|FP|FN|MOTA|MOTP|FPS
|--|--|--|--|--|--|--|--|--|
JDE(with DarkNet53 backbone)|45.0| 48.7 |**91.4**|**5777** | 64672|42.4|0.235|17.84
JDE(with Swin-T backbone)|**48.2**| **54.6** | 88.7 | 8784| **57202**|**45.9**|**0.249**|**23.55**

***Training details:***
JDE with Swin-T backbone is trained with:

 - Swin-T ImageNet pretrained model
 - Half train dataset, 27seqs
 - batch size=32,
 - optimizer AdamW, init lr=3e-4
 - 40Epochs, test with the best mAP model during training(which is 33rd epoch), lr x 0.1 at 31st epoch and 37th epoch(follow Swin Transformer paper)
 - 2 Tesla A100 GPUs, about 5 hours
 
 JDE with DarkNet is similar.

***Trained model***:  
Baidu Link: [link](https://pan.baidu.com/s/1iU9GNoc1IDG_4PFl7kXWiQ)  

code：ngm1

**TODO:**  
I will train MOT17 dataset to compare with DarkNet again   
and I will try to reach better result on VisDrone.  


----
## 1.Installation

Follow JDE installation is good. My env is:

 - python=3.7.0 pytorch=1.7.0 torchvision=0.8.0 cudatoolkit=11.0
 
 you also need:
 -   [py-motmetrics](https://github.com/cheind/py-motmetrics)  (`pip install motmetrics`)
 -   cython-bbox (`pip install cython_bbox`)
 - opencv

in order to use Swin Transformer, please install mmdetection:

```python
pip install openmim
mim install mmdet
```

## 2.Training
**Firstly you should generate image and annotations path following JDE format(see appendix):**

For VisDrone dataset, you can run:
part train dataset (27 seqs):
```python
part train dataset:
python generate_labels_for_VisDronev2.py --if_certain_seqs
```
full train dataset:
```python
python generate_labels_for_VisDronev2.py
```
generate test dataset path:
```python
python generate_labels_for_VisDronev2.py --split 'VisDrone2019-MOT-test-dev'
```


**Then train**  
> if you want to use the Swin-T pretrained model, please download the model in [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)  
> (choose the Swin-T for Mask RCNN) and **rename it as 'swin_t.pth', and put it in 'weights/'.**  


train with swin backbone:
```python
python train.py --backbone 'swin' --cfg 'cfg/yolov3_1088x608_newanchor3-fullswin.cfg'
```
if you want to train on your own dataset, please modify the anchors in cfg file. 
You can use k-means cluster to choose your anchor size:
```python --if_norm True
python choose_anchors.py choose_anchors.py
```
With multi GPUs:
```python
CUDA_VISIBLE_DEVICES=2,3 python train.py --backbone 'swin' --cfg 'cfg/yolov3_1088x608_newanchor3-fullswin.cfg'
```

## 3.Test

After training, you can test the model by:
```python
python track.py --cfg 'cfg/yolov3_1088x608_newanchor3-fullswin.cfg' --weights 'weights/vis_40Epochs_anchor3_lr3e-4_swin_wd1e-2/best_mAP.pt' --test_visdrone --byte_track --save-images
```
Generally you need to modify the weights path, and if you don't want to use byte track and save images, delete the '--byte_track' and '--save-images'.

**more details**, check *run_JDE.txt*.


***Appendix:***   
JDE annotation format:(see [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT))

![format](https://github.com/JackWoo0831/Swin-JDE/blob/master/imgs/jdeformat.png)


