#!/usr/bin/env bash
output_dir="./outputs/COCO2014"

## coco2017train - refcocog val&test images
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
python tools/train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml \
       OUTPUT_DIR "$output_dir/default_training_mscoco_train2017_ignore_refcocog_v2"\
       SOLVER.CHECKPOINT_PERIOD 30000 \
       SOLVER.IMS_PER_BATCH 16 \
       DATASETS.TRAIN "(\"coco_2017_train\", )" \
       DATASETS.TEST "(\"coco_2017_val\", )"\
       DATALOADER.NUM_WORKERS 8 \
       TEST.EVAL_PERIOD 90000 \
       DATASETS.IGNORE_IMG_PATH './RefSegDatasets/refseg_anno/all_img_name_refcocog.json'


### coco2017train - refcoco&+ val&test images
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
#python tools/train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml \
#       OUTPUT_DIR "$output_dir/default_training_mscoco_train2017_ignore_refcocog_v2"\
#       SOLVER.CHECKPOINT_PERIOD 30000 \
#       SOLVER.IMS_PER_BATCH 16 \
#       DATASETS.TRAIN "(\"coco_2017_train\", )" \
#       DATASETS.TEST "(\"coco_2017_val\", )"\
#       DATALOADER.NUM_WORKERS 8 \
#       TEST.EVAL_PERIOD 90000 \
#       DATASETS.IGNORE_IMG_PATH './RefSegDatasets/refseg_anno/all_img_name_refcoco&+.json'


### coco2017train - refcoco,refcoco+,refcocog val&test images
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
#python tools/train_net.py --num-gpus 8 --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml \
#       OUTPUT_DIR "$output_dir/default_training_mscoco_train2017_ignore_refcocog_v2"\
#       SOLVER.CHECKPOINT_PERIOD 30000 \
#       SOLVER.IMS_PER_BATCH 16 \
#       DATASETS.TRAIN "(\"coco_2017_train\", )" \
#       DATASETS.TEST "(\"coco_2017_val\", )"\
#       DATALOADER.NUM_WORKERS 8 \
#       TEST.EVAL_PERIOD 90000 \
#       DATASETS.IGNORE_IMG_PATH './RefSegDatasets/refseg_anno/all_img_name_vt.json'

