---
title: Delta-GAN-Encoder
date: 2021-11-18 18:08:21
tags: [GAN,面部编辑, 解纠缠]
categories: 论文
---

https://arxiv.org/pdf/2111.08419.pdf

作者：Nir Diamant 

单位：Technion - Israel Institute of Technology

## 摘要

理解和控制生成模型的潜在空间是一项复杂的任务。在本文中，我们提出了一种新的学习方法来控制预训练GAN的潜在空间中的任何期望属性，以便相应地编辑合成的和真实的数据样本。我们进行Sim2Real learning，依靠最小的样本来实现无限数量的连续精确编辑。我们提出了一个基于Autoencoder的模型，该模型学习编码图像之间变化的语义，作为后来编辑新样本的基础，实现精确的期望结果。虽然以前的编辑方法依赖于潜在空间的已知结构(例如，StyleGAN中某些语义的线性)，但我们的方法本质上不需要任何结构约束。我们在面部图像领域演示了我们的方法:编辑不同的表情、姿势和照明属性，实现最先进的结果。



<!--more-->

## motivation

- 语义控制不连续，或者存在部分纠缠
- 真实世界中派生的数据集包含相同的样本，只是不同于特定的属性，这可能有助于精确属性的定义和推断，尽管这几乎是不可能实现的。

## contribution

- 允许编辑图像具有无限的特异性，不受属性纠缠的阻碍，学习隔离不同的语义。
- 学习语义的行为，不需要任何结构约束，允许它在潜在空间中找到编辑的精确非线性路径。

## approach

给定一个图片$A_i$，寻找隐空间$a_j\in W^+$在预训练的生成对抗网络当中得到图片 $A_j$。

首先学习$\Delta_{i,j}$ 一个表示语义的低维矩阵，然后将其结合到$a_i\in W^+$中得到期望的结果。两部分一起学习。

![模型框架](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201035-979609.png)

为了监督学习$\Delta$-s，创建了一个由女性和男性的合成3D模型生成。数据集分为两类，$A_1,A_2,...,A_n$和$B_1.B_2,...,B_n$一个属性延序列变化，而其他属性保持不变。然后将$A_i$和$B_i$投影到隐空间矩阵$a_i$和$b_i$。

模型基于Autoencoder架构，

$\Delta_{i,j}=E(a_i,a_j)$

$\hat{b_j}=D(b_i,\Delta _{i,j})+b_i$

$l_{\text {residual }}=\lambda_{1} \cdot\left\|\mathbf{a}_{j}-\hat{\mathbf{a}_{j}}\right\|_{2}^{2}+\lambda_{2} \cdot\left\|\mathbf{b}_{j}-\hat{\mathbf{b}_{j}}\right\|_{2}^{2}$

前半部分是当前输入图片的损失，后边是其他语义未知图片的偏移的损失。

为了防止过拟合加入噪声$\left\{\begin{array}{l}
\mathbf{a}_{i} \leftarrow \mathbf{a}_{i}+\mathbf{n}_{\mathbf{a}} \\
\mathbf{b}_{i} \leftarrow \mathbf{b}_{i}+\mathbf{n}_{\mathbf{b}}
\end{array}\right.$

$\hat{\mathbf{b}}_{j}=D\left(E\left(\mathbf{a}_{i}+\mathbf{n}_{\mathbf{a}}, \mathbf{a}_{j}+\mathbf{n}_{\mathbf{a}}\right), \mathbf{b}_{i}+\mathbf{n}_{\mathbf{b}}\right)+\mathbf{b}_{i}$

$D\left(\mathbf{a}_{k}, \alpha \cdot \Delta_{i, j}\right)=\hat{\mathbf{a}}_{k+\alpha \cdot(j-i)}$ 下标对应图像偏移

## experiment



![效果](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201037-163564.png)

![FID和CS](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201039-2700.png)

![PCA降维分析变量变化量](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201043-259747.png)

发现其并不是直线变化。

![PCA降维后的空间分布](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201045-246571.png)

$\Delta$-s空间降维结果好，说明解耦效果好。

## Method Limitations

最后，我们展示了几个我们的模型未能实现其确切目标的情况，导致一些纠缠的属性变化。未能达到目标形象可能有几个原因:

- 图像在语义意义上不完美地投射到潜在空间。
- GAN训练数据表达的不寻常性。失败的情况可能表明，一些属性需要比其他属性更费力的努力来解开。

![image-20211118222058251](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201046-449913.png)

![image-20211118222107475](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201048-583008.png)

![image-20211118222112986](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/201049-86545.png)

## 想法

通过将原来的因变量通过encoder降维到$\Delta $-s空间想要实现接纠缠，然后再将$\Delta $-s空间的移动投射到因变量空间。其中的关键点便是得到$\Delta $-s空间的移动，使用了人造合成的图像去引导。结果较差的可能为$\Delta $-s空间也未必是线性。可以参考李群的VAE论文:happy:。

