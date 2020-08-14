#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/6/25 15:20
# @Author  : Yongfei Liu
# @Email   : liuyf3@shanghaitech.edu.cn


import numpy as np
import pickle
import os.path as osp
import os
import spacy
import json


path = './RefSegDatasets/refseg_anno'

def dataset_extraction():



    vocab = []

    dataset = 'refcocog'
    vocab = extract_anno(dataset, 'training_anno', vocab)
    print('complete', dataset, 'training_anno')

    vocab = extract_anno(dataset, 'val_anno', vocab)
    print('complete', dataset, 'val_anno')

    vocab = extract_anno(dataset, 'test_anno', vocab)
    print('complete', dataset, 'test_anno')
    #
    # vocab = extract_anno(dataset, 'testB_anno', vocab)
    # print('complete', dataset, 'testB_anno')

    with open(osp.join(path, dataset, 'vocab.json'), 'w') as dump_f:
        json.dump(vocab, dump_f)



def extract_anno(data, split, vocab):

    with open(osp.join(path, '{}/{}.pkl'.format(data, split)), 'rb') as load_f:
        val_anno = pickle.load(load_f)

    nlp = spacy.load("en_core_web_sm")

    for key_img, value in val_anno.items():
        for ref_id, ref_value in value.items():
            ref_value.pop('segmask')
            sents = ref_value['sents']
            for sent_id, sent in sents.items():
                raw_st = sent['sent']
                tokens = sent['tokens']
                noun_phr, nouns = spacy_extraction(nlp, raw_st)
                sent['nouns'] = nouns
                sent['noun_phr'] = noun_phr

                for tk in tokens:
                    if tk not in vocab:
                        vocab.append(tk)

    path1 = osp.join(path, '{}/{}_lite.pkl'.format(data, split))
    print('save', path1)
    with open(path1, 'wb') as load_f:
        pickle.dump(val_anno, load_f)

    return vocab


def spacy_extraction(nlp, sent):

    # Load English tokenizer, tagger, parser, NER and word vectors
    doc = nlp((sent))
    # Analyze syntax
    noun_phr = [chunk.text for chunk in doc.noun_chunks]
    nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    return noun_phr, nouns


if __name__ == '__main__':
    dataset_extraction()