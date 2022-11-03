---
title: NÜWA Visual Synthesis Pre-training for Neural visual world creation
date: 2021-11-28 14:22:55
tags: [生成模型]
categories: 论文
---

## 3D Data Representation

unified 3D notation $X\in \Bbb{R}^{h\times w\times s\times d}$，$h$ and $w$ denote the number of tokens in spatial axis $s$ denotes the number of tokens in the temporal axis , $d$ is the dim of each token.

- text use lower-cased byte pair encoding (BPE) to token and embed them to $\Bbb{R}^{1\times 1\times s\times d}$ 

- images  using 1D VQ-GAN to encode to $\Bbb{R}^{h\times w\times 1\times d}$ 

- video using 2D VQ-GAN to encode to $\Bbb{R}^{h\times w\times s\times d}$ 

## 3D Nearby Self-Attention

a coordinate $(i,j,k)$ under $X$ ,$(\lfloor i \frac{h'}{h} \rfloor,\lfloor j \frac{w'}{w} \rfloor,\lfloor k \frac{s'}{s} \rfloor)$,  a width, height and temporal extent $e^w,e^h,e^s\in \Bbb{R}^+$.
$$
N^{(i,j,k)}=\left\{ C_{abc}| \left|a-i' \right|\le e^h,\left|b-j' \right|\le e^w,\left|c-k' \right|\le e^s\right\}\in \Bbb{R}^{e^h\times e^w\times e^s \times d^{in}}
$$


$$
Q^{(i,j,k)}=XW^Q\\K^{(i,j,k)}=N^{(i,j,k)}W^K\\V^{(i,j,k)}=N^{(i,j,k)}W^V\\y_{ijk}=softmax(\frac{(Q^{(i,j,k)})^TK^{(i,j,k)}}{\sqrt{d^{in}}})V^{(i,j,k)}
$$



## 3D Encoder-Decoder

$$
Y_{i j k}:=Y_{i j k}+P_{i}^{h}+P_{j}^{w}+P_{k}^{s}\\
C_{i j k}:=C_{i j k}+P_{i}^{h^{\prime}}+P_{j}^{w^{\prime}}+P_{k}^{s^{\prime}}
$$

$$
C^{(l)}=3DNA(C^{(l-1)},c^{(l-1)})
$$

$$
Y^{(l)}_{ijk}=3DNA(Y^{l-1}_{<i,<j,<k},Y^{l-1}_{<i,<j,<k})+3DNA(Y^{l-1}_{<i,<j,<k},C^{L})
$$

inital $V^{(1)}_{0,0,0}$ is a special `< bos >` token learned



![image-20211128160612935](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202111/28/162124-463286.png)

## Implementation Details

- text $(e^w,e^h,e^s)=(1,1,\infty)$ $1,1,77,1280$
- image $(e^w,e^h,e^s)=(3,3,1)$ $21,21,1,1280$
- video $(e^w,e^h,e^s)=(3,3,3)$ $21,21,10,1280$

We pre-train on **64 A100 GPUs** for **two weeks** with the layer $L$ in  set to 24, an Adam optimizer with a learning rate of 1e-3, a batch size of 128, and warm-up 5% of a total of 50M steps. The final pre-trained model has a total number of 870M parameters.
