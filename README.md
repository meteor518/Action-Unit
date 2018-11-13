# Smile2pay

## AU Introduction

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

* `data`：存放训练与验证数据
  * `proprocess.ipynb`：分割数据
* `model`：存放预训练模型
  * `model-y1-test2`：InsightFace mobileface training on MS1M
  * `model-mob-al`：InsightFace mobileface -> Fine tuning on EmotioNet 2w5 clean data -> Active Learning on EmotioNet 90w noisy training data
* `output`：存放训练完成的模型
  * `model_last`：mob_al with [m==0.3 s==64 arcloss]
  * `model_6`：mob_al with [m==0.3 s==64 arcloss] + [alpha==0.5 gamma==2 focal loss]
* `src`：网络训练代码
* `test`：网络测试代码

## Preprocessing

* `Alignment`

  **InsightFace**默认的5个Alignment Coordinates如右下图所示，调整为左下图（padding==0.15），使得**AU肌肉信息**在图片中的占比增大，减少不相关的区域

  <figure class="half">
      <img src="./document/dlib.png" width="40%">
      <img src="./document/insightface.png" width="40%">
  </figure>

* `Split`

  * 数量小于**30**的类别直接当作**distractor**集
  * 8k+AU全0样本随机丢弃**6k**
  * 对每类别各抽样**0.1**份作为验证集，最后会全量训练

## Loss function

* `ArcLoss`：[InsightFace](https://github.com/FlareActor/insightface)

  margin_m取**0.3**最佳，更大时（如0.5）training更加苛刻，导致train acc更低，loss更高；在验证集上也效果较差

* `Focal Loss`：由于不平衡，遂采用`[alpha==0.5 gamma==2 focal loss]`降低多数类对loss的权重，使网络集中精力优化少数的困难类

## Metric

* SIDE (Same Identity Different Expression)

* SEDI (Same Expression Different Identity)

  <figure class="half">
      <img src="./document/model_1.png" width="40%">
      <img src="./document/model_6.png" width="40%">
  </figure>

* tpr & fpr of validation data：[腾讯文档](https://docs.qq.com/sheet/DRFdFYmNLSmh4SXVw?opendocxfrom=admin&tab=BB08J2)

## Reference

1.[FACS - Facial Action Coding System](https://www.cs.cmu.edu/~face/facs.htm)

2.[Facial Action Coding System (FACS) – A Visual Guidebook](https://imotions.com/blog/facial-action-coding-system/)

3.[面部编码系统-微表情](https://wenku.baidu.com/view/298d8a8a19e8b8f67c1cb969.html)