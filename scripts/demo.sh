#!/usr/bin/env bash

python demo/demo_run_imgs.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml \
       --opts MODEL.WEIGHTS './outputs/COCO2014/default_training_mscoco_train2017_ignore_vt_v1/model_0269999.pth'