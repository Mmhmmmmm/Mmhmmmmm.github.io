---

title: Taming Transformers for High-Resolution Image Synthesis
date: 2021-12-16 15:37:36
tags: [生成算法,Transformer,codebook]
categories: 论文
---

![image-20211216155217239](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/155233-854490.png)

## Abstract

设计学习远程交互顺序数据，变压器继续显示最先进的结果，各种任务。与cnn不同的是，它们不包含优先考虑本地互动的归纳偏见。这使得它们具有表达能力，但在计算上也不适合长序列，比如高分辨率图像。我们演示了如何将cnn的归纳偏差的有效性与变压器的表现力相结合，使它们能够建模，从而合成高分辨率图像。我们展示了如何(i)使用cnn学习图像成分的context-rich词汇表，反过来(ii)利用transformer在高分辨率图像中有效地建模它们的成分。我们的方法很容易应用于条件合成任务，其中非空间信息(如对象类)和空间信息(如分割)都可以控制生成的图像。特别地，我们给出了第一个用变换进行语义引导的百万像素图像合成的结果，并获得了类条件ImageNet上自回归模型的最新进展。

##  Approach

以前的工作[55,8]是将Transformer应用于图像生成，对于大小为64×64pixels的图像，结果很有希望，但由于序列长度的成本是二次增加的，不能简单地缩放到更高的分辨率。高分辨率图像合成需要一个模型，该模型能够理解图像的组成，使其能够生成局部真实以及全局一致的模式。因此，我们不使用像素表示图像，而是将其表示为一个码本中感知丰富的图像成分的组合。通过学习一种有效的代码，如第3.1节所述，我们可以显著地减少合成的描述长度，这使得我们可以有效地利用第3.2节所述的变压器架构在图像中建模它们的全局相互关系。这种方法，如图2所示，能够在无条件和条件设置下生成真实和一致的高分辨率图像。

![image-20211216160416427](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/160416-53601.png)

###  Learning an Effective Codebook of Image Constituents for Use in Transformers

- image $x\in \Bbb{R}^{h\times w\times 3}$ to $z_q\in \Bbb{R}^{h\times w\times n_z}$,$n_z$ is the dim of codes .
- codebook $Z={z_k}_{k=1}^K$ 

- $\hat{x}=G(z_q)$
- $\hat{z}=E(x)$
- $z_q=q(\hat{z}):=(\arg{min}_{z_k\in Z}||\hat{z}_{i,j}-z_k||)$
- loss $L_{VQ}(E,G,Z)=||x-\hat{x}||^2+||sg[E(x)]-z_q||^2_2+||sg[z_q]-E(x)||^2_2$, $sg[]$ denotes stop-gradient operation



#### Learning a Perceptually Rich Codebook

使用Transformer来表示图像作为潜在图像成分的分布，需要我们推动压缩的极限和学习丰富的码本。为此，我们提出了vqgan，一种原始VQV AE的变体，并使用一个鉴别器和感知损失[40,30,39,17,47]，以在增加压缩率的情况下保持良好的感知质量。需要注意的是，这与之前的工作不同，之前的工作是基于像素[71,61]和基于变压器的自回归模型[8]在一个浅层量化模型之上。更具体地说，我们将[72]中lrecby使用的theL2loss替换为知觉损失，并引入一个基于补丁的判别器[28]的对抗训练程序，目的是区分真实图像和重建图像:

- $L_{GAN}(\{E, G,Z\}, D) = [logD(x) + log(1−D(\hat{x}))]$

- $Q^*=\arg{min}_{E,G,Z}{max}_{D}\Bbb{E}_{x~p(x)}[L_{VQ}(E,G,Z)+\lambda L_{GAN}(\{E,G,Z\},D)]$
- $\lambda=\frac{\nabla_{G_L}[L_rec]}{\nabla_{G_GAN}[L_rec]+\delta}$

### Learning the Composition of Images with Transformers

#### Latent Transformers

Given indices $s<i$, the transformer learns to predict the distribution of possible next indices,

$p(s) =\prod_ip(s_i|s_{<i})$

$L_{Transformer}=\Bbb{E}_{x∼p(x)}[−\log{p(s)}]$

#### Conditioned Synthesis

$p(s) =\prod_ip(s_i|s_{<i},c)$

#### Generating High-Resolution Images

![image-20211216161603421](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/162156-748526.png)

## Experiments

![image-20211216161702818](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/162155-235067.png)

![image-20211216161722969](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/162152-181974.png)

![image-20211216161755213](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/161755-886867.png)

![image-20211216161828226](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/161934-917364.png)

![image-20211216161843590](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/162134-585430.png)

![image-20211216161859596](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/16/162326-87745.png)

![image-20211216161925988](C:\Users\ZJW\AppData\Roaming\Typora\typora-user-images\image-20211216161925988.png)

## VQGAN+CLIP

![img](https://pic2.zhimg.com/v2-3840c8e97af3448fd5c529dfab9f0099_b.jpg)

### X + CLIP

VQGAN+CLIP 只是将图像生成器与 CLIP 相结合的一个例子。但是，可以用任何类型的生成器替换 VQGAN，并且根据生成器的不同，它仍然可以很好地工作。X + CLIP 的许多变体已经出现，例如 StyleCLIP(网址：[https://github.com/orpatashnik/StyleCLIP](https://link.zhihu.com/?target=https%3A//github.com/orpatashnik/StyleCLIP)) （StyleGAN + CLIP）、 CLIPDraw(网址：[https://arxiv.org/abs/2106.14843](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.14843)) （使用矢量艺术生成器）、BigGAN + CLIP 等等。甚至 还有使用音频而不是图像的AudioCLIP (网址：[https://github.com/AndreyGuzhov/AudioCLIP](https://link.zhihu.com/?target=https%3A//github.com/AndreyGuzhov/AudioCLIP))。

![img](https://pic1.zhimg.com/v2-b5784d58c9f081cb9b03b38373d64e28_b.jpg)

图片：使用 StyleCLIP 进行图像编辑



![img](https://pic2.zhimg.com/v2-84af8955dce4afb6440f5207fa06c1f5_b.jpg)

<video src="https://vdn1.vzuu.com/SD/2742466c-d706-11eb-8bab-f2503ac034e9.mp4?disable_local_cache=1&auth_key=1639646596-0-0-ea8095168f978bae6b1b309e3aeae852&f=mp4&bu=pico&expiration=1639646596&v=hw" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

加上“3D photo inpainting”竟然可以生成立体构图

<video src="https://vdn1.vzuu.com/SD/275462ac-d706-11eb-85e2-da1003eb494f.mp4?disable_local_cache=1&auth_key=1639646587-0-0-543317f3d824b504110b69533397a7ef&f=mp4&bu=pico&expiration=1639646587&v=hw" controls="controls" width="500" height="300">您的浏览器不支持播放该视频！</video>

甚至能用来猜猜那些从未露面的大佬们，比如神秘的比特币之父中本聪

![img](https://pic2.zhimg.com/v2-4b7380f37007bd92247ae617bfc82759_b.jpg)



<video class="ztext-gif GifPlayer-gif2mp4" src="https://vdn3.vzuu.com/SD/273a65e6-d706-11eb-bcd2-da8b61ae19d4.mp4?disable_local_cache=1&amp;auth_key=1639646772-0-0-242d526345d82199ec642124f801faee&amp;f=mp4&amp;bu=pico&amp;expiration=1639646772&amp;v=tx" data-thumbnail="https://pic3.zhimg.com/v2-282a872f4ba46f4b46b760d3d89af9d2_b.jpg" poster="https://pic3.zhimg.com/v2-282a872f4ba46f4b46b760d3d89af9d2_b.jpg" data-size="normal" preload="metadata" loop="" playsinline=""></video>

<video class="_1k7bcr7" preload="metadata" playsinline="" webkit-playsinline="" x-webkit-airplay="deny" src="https://vdn1.vzuu.com/LD/bbd326b4-3aea-11ec-8592-66e4ae02c6b3-v4_t111-vlbchuzBoD.mp4?disable_local_cache=1&amp;auth_key=1639646964-0-0-0fef7b84e785ae5552df3de6253fec88&amp;f=mp4&amp;bu=http-com&amp;expiration=1639646964&amp;v=hw" style="object-fit: contain;"></video>

![img](https://pic2.zhimg.com/v2-9a1cdb24655534997792c2aff7c6e7dd_b.jpg)







## VQGAN+transformer

dall-e+VQGAN 

![coco_trained](https://user-images.githubusercontent.com/3994972/125195424-248ac280-e21b-11eb-8231-cd9cede6d549.png)

![cocosamples2](https://user-images.githubusercontent.com/3994972/125196859-12138780-e221-11eb-8c63-a5ffef6615e8.png)

![image](https://user-images.githubusercontent.com/12422441/113150373-f76b6300-926e-11eb-86cc-3d920da2f46b.png)

![image](https://user-images.githubusercontent.com/12422441/113150647-40bbb280-926f-11eb-8975-55968e2afb09.png)



![progress](C:\Users\ZJW\Downloads\progress.png)
