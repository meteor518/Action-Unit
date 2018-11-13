# Action Unit (Multi-label)

## preprocess.py: 数据预处理， 将图片和标签读取，保存为Mxnet读取的rec文件
```shell
python preprocess.py -i ./images/(It's your imags path) -t ./label.txt(your label file) -rec ./data/train.rec(the output .rec file) -s 112
```

## train_with_eval.py: train code
```shell
python train_with_eval.py -rec ./data/train.rec -e ./data/eval.rec -o ./output/ -gpu 0
```
