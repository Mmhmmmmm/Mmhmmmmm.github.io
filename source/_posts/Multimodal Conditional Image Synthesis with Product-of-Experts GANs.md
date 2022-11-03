---
title: Multimodal Conditional Image Synthesis with Product-of-Experts GANs
date: 2021-12-14 15:57:15
tags: [生成模型,text to image]
categories: 论文
---

[Multimodal Conditional Image Synthesis with Product-of-Experts GANs (deepimagination.cc)](https://deepimagination.cc/PoE-GAN/)

## Abstract

现有的条件图像合成框架基于单一模式的用户输入生成图像，例如文本、分割、草图或样式参考。当可用时，它们通常无法利用多模态用户输入，这降低了它们的实用性。为了解决这一限制，我们提出了 Product-of-Experts Generative Adversarial Networks (PoE-GAN) 框架，该框架可以合成基于多个输入模式或其中任意子集(甚至空集)的图像。PoE-GAN由 product-of-experts  generator和多模态多尺度投影鉴别器组成。通过我们精心设计的训练方案，PoE-GAN学会了合成高质量和多样性的图像。除了在多模态条件图像合成方面的进步，PoE-GAN在单模态环境下的测试结果也优于现有的最佳单模态条件图像合成方法。

## Product-of-experts GANs

- images $x$ paired with $M$ different input modalities $(y_1,y_2,...,y_M)$ 

### Product-of-experts modeling

....
