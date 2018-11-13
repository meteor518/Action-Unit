# Action Unit

## Introduction

* `AU1`：Inner Brow Raiser（眉毛内角抬起）

* `AU2`：Outer Brow Raiser （眉毛外角抬起）
* `AU4`：Brow Lowerer（皱眉或降低眉毛）
* `AU5`：Upper Lid Raiser（上眼睑上升）
* `AU6`：Cheek Raiser（脸颊提升）
* `AU9`：Nose Wrinkler（皱鼻）
* `AU12`：Lip Corner Puller（倾斜向上拉动嘴角）
* `AU17`：Chin Raiser（下唇向上）
* `AU20`：Lip stretcher（嘴角拉伸）
* `AU25`：Lips part（张嘴并指定双唇分离的长度）
* `AU26`：Jaw Drop（张嘴并指定颌部下降的距离）
* `AU43`：Eyes Closed（闭眼）

## model
* model_y1_test2： mobileface的预模型
* model_v4：以样本数大于50的所有类作为train set，m=0.5，batch_size=512，在insight face alignment处理的图片上训练的模型。

## data_process: 数据处理
* count.ipynb：数据类别统计和分类
* get_128d.py：利用已训练模型进行特征提取
* tsne_feature.ipynb：利用TSNE将特征降维显示
* acc.py： 计算top-1
* tprfpr.py： 计算tpr、fpr

## src: train code

## Reference

1.[FACS - Facial Action Coding System](https://www.cs.cmu.edu/~face/facs.htm)

2.[Facial Action Coding System (FACS) – A Visual Guidebook](https://imotions.com/blog/facial-action-coding-system/)

3.[面部编码系统-微表情](https://wenku.baidu.com/view/298d8a8a19e8b8f67c1cb969.html)