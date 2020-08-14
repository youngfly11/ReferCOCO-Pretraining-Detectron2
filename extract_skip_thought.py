#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 22:41
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn



import numpy as np
import os.path as osp
import os
import pickle
from collections import OrderedDict
import torch
import json
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES


def check_objects_vocab():

    with open('./flickr30k_datasets/objects_vocab.txt', 'r') as load_f:
        object_vocab = load_f.readlines()

    with open('./flickr30k_datasets/skip-thoughts/dictionary.txt', 'r') as load_f:
        skip_dict = load_f.readlines()

    object_v1 = []
    for vocab in object_vocab:
        object_v1.append(vocab.strip())

    skip_v1 = []
    for sk_dict in skip_dict:
        skip_v1.append(sk_dict.strip())

    for vocab in object_v1:

        vocab = vocab.split(' ')
        for vo in vocab:
            if vo not in skip_v1:
                print(vocab)


def _make_emb_state_dict(self, dictionary, parameters):
    weight = torch.zeros(len(self.vocab)+1, 620) # first dim = zeros -> +1
    unknown_params = parameters[dictionary['UNK']]
    nb_unknown = 0
    for id_weight, word in enumerate(self.vocab):
        if word in dictionary:
            id_params = dictionary[word]
            params = parameters[id_params]
        else:
            print('Warning: word `{}` not in dictionary'.format(word))
            params = unknown_params
            nb_unknown += 1
        weight[id_weight+1] = torch.from_numpy(params)
    state_dict = OrderedDict({'weight':weight})
    if nb_unknown > 0:
        print('Warning: {}/{}({}) words are not in dictionary, thus set UNK embedding parameter to init'
              .format(nb_unknown, len(self.vocab), len(dictionary)))
    return state_dict


def extract_embedding():
    # {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},


    path = './RefSegDatasets/refseg_anno'
    dataset = 'refcoco+'


    # object_vocab = []
    # for cat in COCO_CATEGORIES:
    #     thing = cat['isthing']
    #     if thing==1:
    #         object_vocab.append(cat['name'])

    # with open('./flickr30k_datasets/objects_vocab.txt', 'r') as load_f:
    #     object_vocab = load_f.readlines()

    with open('./flickr30k_datasets/skip-thoughts/dictionary.txt', 'r') as load_f:
        skip_dict = load_f.readlines()

    skip_dict = {word.strip():idx for idx, word in enumerate(skip_dict)}

    path_params = './flickr30k_datasets/skip-thoughts/utable.npy'
    params = np.load(path_params, encoding='latin1', allow_pickle=True)  # to load from python2

    # object_embed = []

    # for vocab in object_vocab:
    #     vocab = vocab.strip().split(' ')
    #     vocab_eb = []
    #     for vb in vocab:
    #         vb_idx = skip_dict.get(vb)
    #         vocab_eb.append(params[vb_idx].squeeze())
    #
    #     vocab_eb = np.stack(vocab_eb, axis=0).mean(0)
    #     object_embed.append(vocab_eb)
    #
    # object_embed = np.array(object_embed) ## 1600*620
    # print('object_dim', object_embed.shape)
    #
    # with open(osp.join(path, dataset, 'skip_label.pkl'), 'wb') as pickle_dump:
    #     pickle.dump(object_embed, pickle_dump)

    vocab_file = open(osp.join(path, dataset, 'vocab.json'))
    vocab = json.load(vocab_file)
    vocab_file.close()
    # add_vocab = ['relate', 'butted']
    # vocab.extend(add_vocab)


    skip_thoughts_dict = {}
    for vb in vocab:
        vb = vb.strip()
        vb_idx = skip_dict.get(vb)

        if vb_idx is not None:
            skip_thoughts_dict[vb] = params[vb_idx].squeeze()
        else:

            vb_split = vb.split('-')
            vb_split_embed = []
            for vbs in vb_split:
                vbs_idx = skip_dict.get(vbs)
                if vbs_idx is not None:
                    vb_split_embed.append(params[vbs_idx].squeeze())
                else:
                    print(vb, 'not in dictionary')
                    break
            if len(vb_split_embed) == len(vb_split):
                # print(vb, 'are in list')
                vb_split_embed = np.stack(vb_split_embed, axis=0).mean(0)
                skip_thoughts_dict[vb] = vb_split_embed

    print(len(vocab))
    with open(osp.join(path, dataset, 'skip_vocab.pkl'), 'wb') as pickle_dump:
        pickle.dump(skip_thoughts_dict, pickle_dump)


if __name__ == '__main__':
    extract_embedding()
