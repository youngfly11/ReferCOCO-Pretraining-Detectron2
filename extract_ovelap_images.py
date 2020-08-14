#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/6/24 23:25
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import numpy as np
import os.path as osp
import os
import json



def extract_val_label():

    path = './RefSegDatasets/refseg_anno'

    img_list = []

    # img_list = read_dataset(path, 'refcoco', 'training', img_list)
    # img_list = read_dataset(path, 'refcoco', 'val', img_list)
    # img_list = read_dataset(path, 'refcoco', 'testA', img_list)
    # img_list = read_dataset(path, 'refcoco', 'testB', img_list)
    # print('there are {} images in total'.format(len(img_list)))
    #
    # img_list = read_dataset(path, 'refcoco+', 'training', img_list)
    # img_list = read_dataset(path, 'refcoco+', 'val', img_list)
    # img_list = read_dataset(path, 'refcoco+', 'testA', img_list)
    # img_list = read_dataset(path, 'refcoco+', 'testB', img_list)
    # print('there are {} images in total'.format(len(img_list)))

    img_list = read_dataset(path, 'refcocog', 'training', img_list)
    img_list = read_dataset(path, 'refcocog', 'val', img_list)
    print('there are {} images in total'.format(len(img_list)))

    with open(osp.join(path, 'all_img_name_refcocog.json'), 'w') as dump_f:
        json.dump(img_list, dump_f)





def read_dataset(path, split, data_name, img_list):


    with open(osp.join(path, '{}/{}_list.txt'.format(split, data_name)), 'r') as read_f:
        dataset = read_f.readlines()

    for idx, im_name in enumerate(dataset):
        im_name = im_name.strip().split(',')[0]

        if im_name not in img_list:
            img_list.append(im_name)

    return img_list


if __name__ == '__main__':
    extract_val_label()