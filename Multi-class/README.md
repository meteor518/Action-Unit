# Action Unit (Multi-class)

## data_process: 数据处理
* count.ipynb：数据类别统计和分类
* get_128d.py：利用已训练模型进行特征提取
* tsne_feature.ipynb：利用TSNE将特征降维显示
* acc.py： 计算top-1
* tprfpr.py： 计算tpr、fpr
* preprocess.py：数据预处理，将图片和标签读取，保存为Mxnet读取的rec文件
```shell
python preprocess.py -i ./images/(It's your imags path) -t ./label.txt(your label file) -rec ./data/train.rec(the output .rec file) -s 112
```

## train: train code
```shell
python train.py -num 22(类别数) -rec ./data/train.rec -e ./data/eval.rec -o ./output/ -gpu 0
```
