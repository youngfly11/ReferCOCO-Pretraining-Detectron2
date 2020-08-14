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
| ResNet101-C4 | COCO2017train | refcoco&refcoco+ val&test | 41.01/60.34/44.18|35.33/56.98/37.49|[ckpt_final](https://drive.google.com/file/d/1JD5MyfMyE1CGhR0TQl9BKqyAoeuNpPLh/view?usp=sharing)|[log](https://drive.google.com/file/d/1EwhXW25QVrahZJMnEaH8rBy6XkldJ-TL/view?usp=sharing)|
| ResNet101-C4 | COCO2017train | refcocog val&test         | 40.99/60.15/44.15|35.33/57.03/37.75|[ckpt_final](https://drive.google.com/file/d/1e-xKFl6eZv5VI1CW6sMFl-OEVh3at0L3/view?usp=sharing)|[log](https://drive.google.com/file/d/1sEqIoZbwOAQbXGngmmHD5ahT9BfvTWpv/view?usp=sharing)|
| ResNet101-C4 | COCO2017train | refcoco&+&g val&test      | 42.08/61.60/45.45|36.36/58.35/38.89|[ckpt_final](https://drive.google.com/file/d/1MryfhLe71pt1gpvh5G_XigIecIauabjT/view?usp=sharing)|[log](https://drive.google.com/file/d/1m45FnXDQUvwg0lOsQW8gkcNMD2Qcxc77/view?usp=sharing)|


**MSCOCO-refervaltest** denotes MSCOCO2017 training images minus the refcoco, refcoco+, refcocog val&test images


## Prerequisites

* Detectron2 Compiler    GCC 7.4
* PyTorch                1.4.0+cu100
* Pillow                  7.1.2
* cv2                    4.2.0

## Installation

See [INSTALL.md](INSTALL.md).

