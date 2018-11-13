# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 10:07:19 2018

@author: lmx
"""

# 得到128维特征
import mxnet as mx
import mxnet.gluon.data.vision.transforms as T
from mxnet import gluon, nd, autograd
from tqdm import tqdm
from train_with_eval import Transpose, TransposeChannels

import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--checkpoint', '-model', default='../model_ziqi/v4_1/checkpoint/model-0100.params',
                        help='path of .params file of model')
    parser.add_argument('--gpu-device', '-gpu', type=int, required=True, help='specify gpu id')
    
    parser.add_argument('--record', '-rec', default='../Data/data_ziqi/rec/train_above50_newlabel.rec',
                        help='path of .rec file of target data')
    parser.add_argument('--out-dir', '-out', default='../Data/data_ziqi/npy/train/',
                            help='save path of .npy file of target features and labels')

    args = parser.parse_args()

    checkpoint =args.checkpoint
    ctx = mx.gpu(args.gpu_device) if args.gpu_device else mx.cpu()
    
    symbol_file = checkpoint[:-11] + 'symbol.json'
    net = gluon.SymbolBlock.imports(symbol_file, ['data'], checkpoint, ctx)
    
    record = args.record
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    transform = T.Compose([T.Resize(112), Transpose()])
    features = []
    labels_ori = []
    labels = []
    dataloader = gluon.data.DataLoader(
        dataset=gluon.data.vision.ImageRecordDataset(
            filename=record,
            transform=lambda x, y: (transform(x), y)
        ),
        batch_size=1,
        shuffle=False,
        last_batch='keep'
    )
    
    for x, y in tqdm(dataloader, desc='Evaluating', leave=False):
        x = x.as_in_context(ctx)
        y = y.as_in_context(ctx)
        
        f, pred = net(x)
        f = normalize(f.asnumpy())
        features.append(f)
        
        y = y.asnumpy()
        y = np.array(y, dtype=np.int)
        labels_ori.append(y)
        
        valid = np.not_equal(y, 999)
        label = np.bitwise_and(valid, y)
        labels.append(label)
        
    np.save(out_dir + 'features.npy', features)
    labels_ori = np.squeeze(labels_ori, axis=1)
    labels_ori = pd.DataFrame(labels_ori)
    
    labels = np.squeeze(labels, axis=1)
    labels = pd.DataFrame(labels)
    labels_ori.to_csv(out_dir + 'label.csv', index=False)
    labels.to_csv(out_dir + 'label_no999.csv', index=False)