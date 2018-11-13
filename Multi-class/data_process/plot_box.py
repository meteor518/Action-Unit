# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 09:44:12 2018

@author: lmx
"""

import seaborn as sns
import pandas as pd
import numpy as np

f1_path = '../test_data/data1.npy'
f2_path = '../test_data/data2.npy'

f1 = np.load(f1_path)
f2 = np.load(f2_path)

f_frown_1 = f1[0:len(f1):2]
f_frown = f_frown_1
f_frown_2 = f2[:5]
f_frown = np.concatenate([f_frown_1, f_frown_2])

f_laugh_1 = f1[1:len(f1):2]
f_laugh = f_laugh_1
f_laugh_2 = f2[5:]
f_laugh = np.concatenate([f_laugh_1, f_laugh_2])

# 同一个人两个表情间的距离
dist_inner = []
for i in range(len(f_frown)):
    dist_inner.append(np.linalg.norm(f_frown[i]-f_laugh[i]))

#同一种表情间的距离
dist_frown = []
for i in range(len(f_frown)-1):
    dist_frown.extend(np.linalg.norm(f_frown[i]-f_frown[i+1:], axis=1))
    
dist_laugh = []
for i in range(len(f_laugh)-1):
    dist_laugh.extend(np.linalg.norm(f_laugh[i]-f_laugh[i+1:], axis=1))

dist_ex = np.concatenate([dist_frown, dist_laugh])

S = pd.Series(dist_inner)
I = pd.Series(dist_ex)
df = pd.concat([S, I], axis=1)
sns.boxplot(data=df)

    
dist_inner = pd.DataFrame(dist_inner)
dist_frown = pd.DataFrame(dist_frown)
dist_laugh = pd.DataFrame(dist_laugh)

dist_inner.to_csv('../test_data/v2_inner.csv', index=False)
dist_frown.to_csv('../test_data/v2_frown.csv', index=False)
dist_laugh.to_csv('../test_data/v2_laugh.csv', index=False)
    