## Object Detection for Refcoco&Refcoco+&Refcocog

In current referring expression tasks, most of work take the [maskrcnn](https://github.com/lichengunc/mask-faster-rcnn) 
pretrained on **MSCOCO-refervaltest** dataset as object detector to extract a set of object proposals. In this setting,
the object detector can provide nearly ground-truth proposal during training stage. 

In additional, the repo [maskrcnn](https://github.com/lichengunc/mask-faster-rcnn) provide by lichengunc is out-of-data, which was written by python2 and low PyTorch
version. To facilitate the development of the referring expression reasoning community, we provide the new pretrained object
detector model based on the **Detectron2**. I hope this will help all researchers to develop faster, accurate referring reasoning system.


## Performance 

| Backbone     | training set  |  excluded images           | Box-AP/AP50/AP75 | mask-AP/AP50/AP75|Model|Log|
|  :----:      | :----:        |:----:                     | :---:            |:---:|:---:|:---:|
| ResNet101-C4 | COCO2017train | refcoco&refcoco+ val&test | 41.01/60.34/44.18|35.33/56.98/37.49|[ckpt_final]()|[log](./outputs/COCO2014/default_training_mscoco_train2017_ignore_refcoco&+_v1/log.txt)|
| ResNet101-C4 | COCO2017train | refcocog val&test         | 40.99/60.15/44.15|35.33/57.03/37.75|[ckpt_final]()|[log](./outputs/COCO2014/default_training_mscoco_train2017_ignore_refcocog_v1/log.txt)|
| ResNet101-C4 | COCO2017train | refcoco&+&g val&test      | 42.08/61.60/45.45|36.36/58.35/38.89|[ckpt_final]()|[log](./outputs/COCO2014/default_training_mscoco_train2017_ignore_refcoco&+_vt/log.txt)|


**MSCOCO-refervaltest** denotes MSCOCO2017 training images minus the refcoco, refcoco+, refcocog val&test images


## Prerequisites

* Detectron2 Compiler    GCC 7.4
* PyTorch                1.4.0+cu100
* Pillow                  7.1.2
* cv2                    4.2.0

## Installation

See [INSTALL.md](INSTALL.md).

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md),
or the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5).

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.