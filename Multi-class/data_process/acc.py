# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:50:21 2018

@author: lmx
"""

import numpy as np
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    parser.add_argument('--target', '-t', default='../new_npy_m0.5/target/features.npy',
                        help='path of .npy file of target features')
    parser.add_argument('--target-label', '-label', default='../new_npy_m0.5/target/label_no999.csv',
                        help='path of .csv file of target data labels')
    parser.add_argument('--distractor', '-d', default='../new_npy_m0.5/distractor/features.npy',
                            help='path of .npy file of distractor features')
    parser.add_argument('--target-class', '-c', default='../tongji/new_target_classes.csv',
                            help='path of .csv file of target classes')
    args = parser.parse_args()
    
    target_features = np.load(args.target)  
    target_features = np.squeeze(target_features, axis=1)
    distractor_data = np.load(args.distractor) 
    distractor_data = np.squeeze(distractor_data, axis=1) 
    
    label = pd.read_csv(args.target_label)
    label = np.array(label)
    classes = pd.read_csv(args.target_class)
    classes = np.array(classes)
    
    # 将target feature按类组合  
    target_data = []
    for i in range(len(classes)):
        data = []
        for j in range(len(label)):
            if (label[j] == classes[i]).all():
                data.append(target_features[j])
        target_data.append(data)

    
    # 正样本对， 同一类中任取两两配对
    pos = []      # 保存正样本对
    pos_dist = [] # 保存正样本对间距离
    for i in range(len(target_data)):
        # 每一类内随机挑2个配
        every_class = target_data[i]
        for j in range(len(every_class) - 1):
            for k in range(j + 1, len(every_class)):
                pos.append([every_class[j], every_class[k]])
                dist = np.linalg.norm(every_class[j] - every_class[k])
                pos_dist.append(dist)
    print('pos: ', np.shape(pos), np.shape(pos_dist))
    
    
    # 正样本对与distractor计算距离
    pos_dis_data = [] # 保存所有正样本对与distractor所有的距离
    num = 0
    tp = 0
    for i in range(len(pos)):
        # 正样本对中挑选其中一个与distractor中所有去计算距离
        d2 = [] # 保存每个对中的一个与distractor所有计算的距离
        d1 = pos_dist[i]
        
        for j in range(2):
            f1 = pos[i][j] #每个对中的一个距离
            # 与distractor中所有去配对
            dist = np.linalg.norm(f1 - distractor_data, axis=1)
            d2.extend(dist) 
            print(sum(d1>np.array(d2, dtype=np.int32)))
            if sum(d1>np.array(d2, dtype=np.int32)) == 0:
                tp += 1
            num += 1
        if i%100==0:
            print('tp, num:', tp, num)
    
    acc = tp/num
    print('acc: ', acc)
