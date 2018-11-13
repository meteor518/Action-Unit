# Action Unit(AU)

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

## Project Structure
基于`Mxnet`框架实现。

* `model`：存放预训练模型
  * `model-y1-test2`：InsightFace mobileface training on MS1M
  
* `Multi_class`：将每种AU组合看做一个类别，以[Arcface](https://arxiv.org/abs/1801.07698)为loss function，进行多分类的网络训练，以mobilefacew为架构，基于预模型fine-tune。

* `Multi_label`：共12种AU识别，每个AU都是一个二分类标签，进行多标签的网络训练，以mobileface为架构，基于预模型fine-tune。


## Reference

1.[FACS - Facial Action Coding System](https://www.cs.cmu.edu/~face/facs.htm)

2.[Facial Action Coding System (FACS) – A Visual Guidebook](https://imotions.com/blog/facial-action-coding-system/)

3.[面部编码系统-微表情](https://wenku.baidu.com/view/298d8a8a19e8b8f67c1cb969.html)
