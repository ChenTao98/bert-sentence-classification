# 语言模型句子分类

使用预训练语言模型实现句子分类的微调与测试，目前实现的语言模型包括：bert、roberta、albert、gpt2，可以扩展gpt、gpt2等模型

## 环境配置（python=3.7.11）

+ transformers==4.11.3
+ torch==1.9.1

若有其他需要包，使用pip安装即可

## 项目结构

```
data    存放数据的文件夹
model   保存训练模型的文件夹
predict_result  保存预测结果的文件夹
*.py    主要python文件
model_config.json   模型配置文件，配置模型的文件初始权重路径等信息
```

## 数据说明
在data文件夹下包含三个文件, 文件内容每行如下
```
label   text
```
label和text之间使用制表符(\t)分隔

本项目使用数据是[AG's News](https://github.com/srviest/char-cnn-text-classification-pytorch),并将其中的title和description合并成text

## 运行
！！ 运行前请设置model_config.json中的模型文件等配置信息

训练
```
python train.py --model $model_name$ --save_model_pah $save_model_name$ --train
```
若需要测试，则不需要设置参数 --train，具体参数设置可以查看train.py文件中的说明
## 结果(默认参数)

| model | valid_acc | test_acc |
| ---- | ---- | ---- |
| bert | 95.03 | 94.76 |
| albert | 94.44 | 94.53 |
| roberta | 94.84 | 95.01 |
| gpt2 | 94.72 | 94.39 |