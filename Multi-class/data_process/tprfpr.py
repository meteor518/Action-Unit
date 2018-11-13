# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 14:01:48 2018

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
    print('target classes:', np.shape(target_data))
    print('distractor samples:', np.shape(distractor_data))
    # 正样本对, 第一列是label，第二列是预测距离标签
    pos_data = []
    threshold = 1
    for i in range(len(target_data)):
        # 每一类内随机挑2个配
        every_class = target_data[i]
        for j in range(len(every_class)-1):
            dist = np.linalg.norm(every_class[j]-every_class[j+1:], axis=1)
            pos_data.extend((dist<threshold).astype(np.int))
    pos_label = np.ones(len(pos_data))

    print('pos: ', np.shape(pos_data), 'pos_label:', np.shape(pos_label))
    # 负样本对
    neg_data = []
    for i in range(len(target_data)):
        # target中挑选一个类
        every_class = target_data[i]
        for j in range(len(every_class)):
            # 挑选类中的一个样本
            f1 = every_class[j] 
            # 与distractor中所有去配对
            dist = np.linalg.norm(f1-distractor_data, axis=1)
            neg_data.extend((dist<threshold).astype(np.int))
            # target中类与类之间去配对，组成负样本
            if i<len(target_data)-1:
                for c in range(i+1, len(target_data)):
                    # 选其他类
                    new_class = target_data[c]
                    dist = np.linalg.norm(f1-new_class, axis=1)
                    neg_data.extend((dist<threshold).astype(np.int))
                
    
    neg_label = np.zeros(len(neg_data))
    print('neg: ', np.shape(neg_data), 'neg_label:', np.shape(neg_label))
    label = np.hstack((pos_label, neg_label))
    data = np.hstack((pos_data, neg_data))
    print('all: ', data.shape, label.shape)
    # 统计tpr fpr
    # 计算混淆矩阵tp/fp/fn/tn
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    tp += np.sum(np.bitwise_and(np.equal(label, 1), np.equal(data, 1)).astype(np.int32), axis=0)
    tn += np.sum(np.bitwise_and(np.equal(label, 0), np.equal(data, 0)).astype(np.int32), axis=0)
    fp += np.sum(np.bitwise_and(np.equal(label, 0), np.equal(data, 1)).astype(np.int32), axis=0)
    fn += np.sum(np.bitwise_and(np.equal(label, 1), np.equal(data, 0)).astype(np.int32), axis=0)
    
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    print('tpr, fpr', tpr,fpr)