# DBNet-Paddle
在Paddle OCR任务中,文本检测模型使用**DBNet(Differentiable Binarization)**
本项目使用的是PaddleOCR中实现文本检测模型，DBNet网络结构和代码
DBNet论文："https://arxiv.org/pdf/1911.08947.pdf"
代码来自PaddlePaddleOCR官方代码："https://github.com/PaddlePaddle/PaddleOCR"
网络结构详细内容来自："https://blog.csdn.net/weixin_43227526/article/details/135024189?spm=1001.2014.3001.5502"


## Introduction

DBNet是一种基于分割的文本检测网络，使用分割网络提供自适应的thresh，用于二值化。

<div align="center">
    <img src=".\images\network.png">
</div>

## 为什么使用DBNet

- 🔨原始的设置阈值的二值化方法是一个阶梯函数，是不可微的，不能参与到网络模型的训练中

- 💥在DBNet中增加了**threshold map来动态生成每一个像素点对应的阈值**，实现二值化。

## DBNet网络结构
<div align="center">
    <img src=".\images\structure.png">
</div>

网络结构中的特征层：

<div align="center">
    <img src=".\images\neural_network.png">
</div>

## 1、backbone

backbone使用的是ResNet18

## 2、neck
## 3、head
`为什么对于**probability map**,需要做一步腐蚀操作：防止不同的文本框粘连在一起。`
