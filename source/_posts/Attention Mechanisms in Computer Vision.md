---
title: Attention Mechanisms in Computer Vision A Survey
date: 2021-11-17 10:48:46
tags: [注意力机制, 综述]
categories: 论文
---

论文地址：[2111.07624.pdf (arxiv.org)](https://arxiv.org/pdf/2111.07624.pdf)

作者：Meng-Hao Guo ，Tsinghua University

## 简介

人类可以自然有效地在复杂的场景中找到突出的区域。在这种观察的推动下，注意机制被引入到计算机视觉中，目的是模仿人类视觉系统的这方面。这种注意机制可以看作是一个基于输入图像特征的动态权值调整过程。注意机制在图像分类、目标检测、语义分割、视频理解、图像生成、三维视觉、多模态任务和自监督学习等视觉任务中取得了巨大的成功。本文综述了计算机视觉中的各种注意机制，并对其进行了分类，如通道注意、空间注意、时间注意和分支注意;一个相关的repositoryhttps://github.com/MenghaoGuo/Awesome-Vision-Attentionsis专门用于收集相关的工作。本文还提出了注意机制研究的未来方向。

<!--more-->

## 介绍

第一阶段 RAM，第二阶段 STN ，第三阶段 CBAM和ECANet，第四阶段 自注意力。 



| Attention category           | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| Channel attention            | Generate attention mask across the channel domain and use it to select important channels. |
| Spatial attention            | Generate attention mask across spatial domains and use it to select important spatial regions (e.g. [15], [61]) or predict the most relevant spatial position directly (e.g. [7], [31]). |
| Temporal attention           | Generate attention mask in time and use it to select key frames. |
| Branch attention             | Generate attention mask across the different branches and use it to select important branches. |
| Channel & spatial attention  | Predict channel and spatial attention masks separately (e.g. [6], [117]) or generate a joint 3-D channel, height, width attention mask directly (e.g. [118], [119]) and use it to select important features. |
| Spatial & temporal attention | Compute temporal and spatial attention masks separately (e.g. [16], [130]), or produce a joint spatiotemporal attention mask (e.g. [131]), to focus on informative regions. |



![attention mechaisms](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/200913-535478.png)

## 计算机视觉中的注意方法



### 一般形式

$$
Attention=f(g(x),x)
$$

其中$g(x)$表示注意力。

以自注意力为例。自注意力可以写为：
$$
Q,K,V={\rm Linear} (x)\\
g(x)={\rm Softmax}(QK)\\
f(g(x),x)=g(x)V\\
$$
以SE 注意力为例。可以写为：
$$
g(x)={\rm Sigmoid}({\rm MLP}({\rm GAP}(x)))\\
f(g(x),x)=g(x)x
$$

### channel attention

![channel attention](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/200910-140477.png)

#### SENet

$$
s=F_{se}(X,\theta)=\sigma(W_2\delta (W_1 {\rm GAP}(X)))\\
Y=sX
$$

由于SE块对计算资源的需求较低，可以在每个剩余单元之后添加一个SE块[145]。但是，SE区块也存在不足。在挤压模块中，全局平均池过于简单，无法捕获复杂的全局信息。在激励模块中，全连通层增加了模型的复杂性。如图4所示，后续工作尝试提高挤压模块(如GSoP-Net[54])的输出，通过改进激励模块(如ECANet[37])来降低模型的复杂性，或者同时改进挤压模块和激励模块(如SRM[55])。

#### GSop-Net

$$
s=F_{gsop}(X,\theta)=\sigma(W\ {\rm RC}({\rm Cov}({\rm Conv}(X))))\\
Y=sX
$$



${\rm Conv(·)}$减少了通道数量，${\rm Cov(·)}$计算协方差矩阵，${\rm RC(·)}$表示行卷积。通过使用二阶池，$GSoP$块提高了在SE块上收集全局信息的能力。然而，这是以额外的计算为代价的。因此，通常在几个残留块之后添加一个$GSoP$块。

#### SRM

$$
s=F_{srm}(X,\theta)=\sigma({\rm BN}({\rm CFC}({\rm SP}(X))))\\
Y=sX
$$



SRM块改善了挤压和激励模块，但可以像SE块一样在每个剩余单元后添加。

#### GCT

$$
s=F_{gct}(X,\theta)={\rm tanh}(\gamma CN(\alpha {\rm Norm}(X)))+\beta\\
Y=sX + X
$$

其中$\alpha$、$beta$、$\gamma$为可训练参数，${\rm Norm(·)}$表示各通道的$L2$-范数。$CN$经过通道正常化。GCT块的参数比SE块少，而且它是轻量级的，可以在CNN的每个卷积层之后添加。

#### ECANet

$$
s=F_{eca}(X,\theta)=\sigma({\rm Conv1D}({\rm GAP}(X)))\\
Y=sX
$$

其中${\rm Conv1D}(·)$表示以k大小的$kernal$为跨信道域的一维卷积，用于模拟局部跨信道交互作用。参数决定交互的覆盖范围，在ECA中，内核大小自适应地从通道维数决定，而不是通过手动调优，使用交叉验证:


$$
k=\psi(C)=\left|\frac{\log _{2}(C)}{\gamma}+\frac{b}{\gamma}\right|_{\text {odd }}
$$

与SENet相比，ECANet有一个改进的激励模块，提供了一个高效有效的块，可以很容易地纳入各种cnn。

#### FcaNet

$$
s=F_{fca}(X,\theta)=\sigma(W_2\delta [([\text{DCT} (\text{Group}(X))])])\\
Y=sX
$$

速度会有较为明显的提升。

#### EncNet

$$
e_k=\sum_{i=1}^N{e^{-s_k\left\|X_i-d_k\right\|^2}(X_i-d_k)}/\sum_{j=1}^K{e^{-s_j\left\| X_i-d_j\right\|^2}}\\
e=\sum_{k=1}^{K}\phi(e_k)\\
s=\sigma (We)\\
Y=sX\\
$$

CEM不仅增强了类相关的特征图，而且通过合并se损失，迫使网络平等地考虑大小物体。由于其轻量级的架构，CEM可以应用于各种框架，而且计算开销很低

#### Bilinear Attention

在GSoP-Net[54]之后，Fang等人[146]声称以前的注意模型只使用一阶信息，而忽略了高阶统计信息。因此，他们提出了一种新的双线性注意块(双注意)来捕获每个通道内的局部成对特征交互，同时保留空间信息。
$$
\tilde{x}=\text{Bi}(\phi(X))=\text{Vec}(\text{UTri}(\phi(X)\phi(X)^T))\\
\hat{x}=\omega(\text{GAP}(\tilde{x}))\varphi(\tilde{x})\\
s=\sigma(\hat{x})\\
Y=sX\\
$$
双注意可以合并到任何CNN主干，以提高其代表性的力量，同时抑制噪声。

#### summary

![image-20211122202438528](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/202439-204980.png)

![image-20211122202453058](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/202454-500343.png)



### Spatial Attention

![image-20211122202555938](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/202556-498709.png)

#### RAM

![ Attention process in RAM](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/22/211548-4899.png)

这提供了一种简单而有效的方法，将网络集中在关键区域，从而减少了网络执行的计算次数，特别是在大输入的情况下，同时提高了图像分类结果。

#### Glimpse Network

$$
g_t=f_{image}(X)\cdot f_{loc}(l_t)\\
r_t^{(1)}=f_{rec}^{(1)}(g_t,r_{t-1}^{(1)})\\
r_t^{(2)}=f_{rec}^{(2)}(r_t^{(1)},r_{t-1}^{(2)})\\
l_{t+1}=f_{emis}(r_t^{(2)})\\
y=f_{cls}(r_t^{(1)})\\
$$

#### Hard and soft attention

$$
e_{t,u}=f_{att}(a_i,h_{t-1})\\
\alpha _{t,i}= \exp{(e_{t,i})}/\sum_{k=1}^L{\exp{(e_{t,k})}}\\
z_t=\sum_{i=1}^L{\alpha_{t,i}\alpha _i}\\
$$

#### Attention Gate

$$
S=\sigma(\varphi(\delta(\phi_x(X)+\phi_g(G))))\\
Y=SX
$$

#### STN

$$
\left[\begin{array}{lll}
\theta_{11} & \theta_{12} & \theta_{13} \\
\theta_{21} & \theta_{22} & \theta_{23}
\end{array}\right]=f_{\text {loc }}(U) \\
\left(\begin{array}{l}
x_{i}^{s} \\
y_{i}^{s}
\end{array}\right)=\left[\begin{array}{lll}
\theta_{11} & \theta_{12} & \theta_{13} \\
\theta_{21} & \theta_{22} & \theta_{23}
\end{array}\right]\left(\begin{array}{c}
x_{i}^{t} \\
y_{i}^{t} \\
1
\end{array}\right)
$$



