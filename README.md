# DBNet-Paddle
在Paddle OCR任务中,文本检测模型使用**DBNet(Differentiable Binarization)**
本项目使用的是PaddleOCR中实现文本检测模型，DBNet网络结构和代码<br>
DBNet论文："https://arxiv.org/pdf/1911.08947.pdf"<br>
代码来自PaddlePaddleOCR官方代码："https://github.com/PaddlePaddle/PaddleOCR"<br>
网络结构详细内容来自："https://blog.csdn.net/weixin_43227526/article/details/135024189?spm=1001.2014.3001.5502"<br>


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

backbone使用的是ResNet18,一系列下采样，提取特征。增加DeformableConvV2<br>
对于原始图片，经过残差卷积操作，提取特征，使用Maxpooling进行下采样，得到原始图片宽高（C2,C3,C4,C5）（1/4,1/8,1/16,1/32）

## 2、neck

构建FPN，主干特征提取网络C2/C3/C4/C5,<br>
C5经过上采样与C4对应像素值相加得到out4；<br>
out4经过上采样与C3对应像素值相加得到out3；<br>
out3经过上采样与C2对应像素值相加得到out3；<br>

C5调整通道数的得到P5<br>
out4调整通道数得到P4<br>
out3调整通道数得到P3<br>
out2调整通道数得到P2<br>

P5经过8次上采样得到原始特征图大小1/2的特征层<br>
P4经过4次上采样得到原始特征图大小1/2的特征层<br>
P3经过2次上采样得到原始特征图大小1/2的特征层<br>
## 3、head

**可微分二值化**


经过neckFPN输出原始图片宽高1/4的特征图，经过卷积和上采样，得到原始图片大小，通道数为1的特征图，`probability`和`thresh`，通过`probability`和`thresh`计算得到`approximate_map`。<br>
可微分二值化，通过thresh_map动态设置每一个像素点的阈值。<br>
1、probability_map生成方法：<br>
将原始文本框的标签做腐蚀操作，收缩量D。<mark style="background-color: #F4A460">为什么对于probability map,需要做一步腐蚀操作：防止两行文本框粘连在一起。</mark><br>
计算公式：<br>
$$D=\frac{A(1-r^2)}{L}$$
A为原始区域的面积，L为原始区域的周长，r为收缩系数，取0.4。<br>

2、Threshold_map计算方式：<br>
对原始文本框分别做腐蚀和膨胀3操作，计算圆环区域中的每一个像素点到文本框边界的距离，做为该点的像素值`value=d/D`，用1减得到归一化后的距离，靠近文本框标签区域像素值接近1，靠近圆环边缘区域像素值接近0.<br>

3、通过`probability`和`thresh`计算得到`approximate_map`


$$\hat{B}_{i,j}=\frac{1}{1+e^{-k(P_{i,j}-T_{i.j})}}$$

