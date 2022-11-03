# ViTGAN: Training GANs with Vision Transformers

![image-20211211215328705](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/11/215330-882056.png)

## Abstract

最近，视觉变形(ViTs)在图像识别方面表现出了有竞争力的性能，同时需要较少的视觉特异性诱导偏差。在本文中，我们研究了这种观测是否可以扩展到图像生成。为此，我们将ViT架构整合到生成式对抗网络(GANs)中。我们观察到现有的GANs正则化方法与自我注意交互不良，导致训练过程中严重的不稳定性。为了解决这个问题，我们引入了一种新的正则化技术来用vit训练gan。根据经验，我们的方法名为vitgan，在CIFAR-10、CelebA和LSUN数据集上实现了与最先进的基于cnn的StyleGAN2相当的性能。

## Related Work

### Vision Transformers

视觉变压器(ViT)是一个无卷积变压器，执行图像分类序列的图像补丁。ViT通过对大规模数据集进行预训练，展示了Transformer架构相对于传统cnn的优势。随后，DeiT通过知识蒸馏和正则化技巧提高了vit的采样效率。MLP- mixer进一步drop自我注意，并取代它由一个MLP混合每位置的功能。我们的工作是第一批在GAN模型中利用视觉变形进行图像生成。

### Generative Transformer in Vision

在GPT-3成功的激励下，一些试点工作使用Transformer通过自回归学习或图像和文本之间的跨模态学习来研究图像生成。这些方法与我们的不同，因为它们将图像生成建模为一个自回归序列学习问题。相反，我们的工作是在生成对抗训练范式中训练视觉变形金刚。最接近我们的作品是TransGAN，呈现了一个基于GAN模型的纯变压器。在提出多任务联合训练和局部初始化以获得更好的训练时，TransGAN忽略了训练稳定性的关键技术，表现远远落后于领先的卷积GAN模型。通过我们的设计，本文首次证明了基于transformer的GAN能够实现与最先进的基于cnn的GAN模型相比的具有竞争力的性能。

## Preliminaries: Vision Transformers (ViTs)

- 2D image $x\in \Bbb{R}^{H\times W\times C}$ flatten to a sequence of image patchs $x_p\in \Bbb{R}^{L\times(P^2C)}$  where $L=\frac{H\times W}{P^2}$ 

- Following BERT, add classification embedding $x_{class}$ to image squence , with added 1D positional embeddings $E_{pos}$ to formulate patch embedding $h_0$

- architecture:

  - $$
    h_0=[x_{class};x_p^1E;x_p^2E;...;x_p^LE]+E_{pos},\\E\in \Bbb{R}^{(P^2C)\times D},E_{pos}\in \Bbb{R}^{(L+1)\times D}
    $$

  - $$
    h_l'=MSA(LN(h_{l-1}))+h_{l-1},l=1,...,L
    $$

  - $$
    h_l=MLP(LN(h_l'))+h_l',l=1,...,L
    $$

  - $$
    y=LN(h_0)
    $$

- 其中注意力部分

  - $$
    Attention_h(X)=softmax(\frac{QK^T}{\sqrt{d_h}})V
    $$

  - $$
    MSA(X)=concat_{h=1}^H[Attention_h(X)]W+b
    $$

## Method

### Regularizing ViT-based discriminator

#### Enforcing Lipschitzness of Transformer Discriminator

最近的工作表明，标注的点积自我注意力层的Lipschitz常数可以是无界的，在vit中违反了Lipschitz连续性。为了加强我们的ViT鉴别器的Lipschitzness，我们将点积相似度替换为欧氏距离，并将self-attention的query和key的投影矩阵的权值绑在一起:
$$
Attention_h(X)=softmax(\frac{d(XW_q,XW_k)}{\sqrt{d_h}})XW_v,where W_q=W_k
$$

#### Improved Spectral Normalization.

我们发现变压器块对李普希茨常数的尺度很敏感，当使用SN时，训练进展非常缓慢。类似地，我们发现R1梯度惩罚在使用基于vitc的鉴别器时削弱GAN训练。[14]表明MLP块的小Lipschitz常数可能导致Transformer的输出压缩为秩1矩阵。为了解决这个问题，我们建议增加投影矩阵的谱范数。我们发现，将每一层的归一化权矩阵与谱范数在初始化时相乘就足以解决这一问题。
$$
\bar{W}_{ISN}(W):=\sigma(W_{init})W/\sigma(W)
$$

#### Overlapping Image Patches

ViT判别器由于其学习能力较强，容易出现过拟合。我们的判别器和生成器使用相同的图像表示，根据预定义的网格 P×P将图像划分为不重叠的补丁序列。这些任意的网格分区，如果不仔细调整，可能会鼓励判别器记住局部线索，并停止为生成器提供有意义的损失。我们使用一个简单的技巧来缓解这个问题，通过允许图像补丁之间的一些重叠。对于patch的每个边界边缘，我们通过像素来扩展它，使有效的patch大小$(P+ 2o)×(P+ 2o)$。

### Generator Design

![Generator Architectures.](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/12/155142-11132.png)
$$
h_0=E_{pos},E_{pos}\in \Bbb{E}^{L\times D}
$$

$$
h_l'=MSA(SLN(h_{l-1},w))+h_{l-1}, l=1,...,L,w\in \Bbb{R}^D
$$

$$
h_l=MLP(SLN(h_l',w))+h_l', l=1,...,L
$$

$$
y=SLN(h_L,w)=[y^1,...,y^L], y^1,...,y^L\in \Bbb{R}^D
$$

$$
x=[x_p^1,...,x_p^L]=[f_\theta(E_{fou},y^1),...,f_\theta(E_{fou},y^L)],  x_p^i\in \Bbb{R}^{P^2\times C},x\in \Bbb{R}^{H\times W\times C}
$$

#### Self-modulated LayerNorm

$$
\operatorname{SLN}\left(\mathbf{h}_{\ell}, \mathbf{w}\right)=\operatorname{SLN}\left(\mathbf{h}_{\ell}, \operatorname{MLP}(\mathbf{z})\right)=\gamma_{\ell}(\mathbf{w}) \odot \frac{\mathbf{h}_{\ell}-\boldsymbol{\mu}}{\sigma}+\beta_{\ell}(\mathbf{w}),
$$

#### Implicit Neural Representation for Patch Generation

我们使用一种隐式神经表示来学习一个patch embeddding $y^i\in \Bbb{R}^D$to patch pixel values $x^i_p\in \Bbb{R}^{P^2\times C}$的连续映射。

当结合傅里叶特征[49]或正弦激活函数[46]时，隐式表示可以将生成样本的空间限制为光滑变化的自然信号的空间。

其中$E_{fou}$为P×P位置的傅里叶编码 ，$f_\theta(.,.)$ 为2-layer MLP

值得注意的是，产生器和判别器可以有不同的图像网格，从而有不同的序列长度。我们发现，当将我们的模型缩放到更高分辨率的图像时，通常只增加鉴别器的序列长度或特征维数就足够了。



## Experiments

### Implementation Details

对于32×32分辨率，我们使用一个4块基于维特的鉴别器和一个4块基于ViT-GAN的生成器。对于64×64分辨率，我们将块的数量增加到6。vi - small[15]之后，所有Transformer块的输入/输出特征维数为384，而MLP隐藏维数为1536。不像[15]，我们选择了6个注意力头。我们发现增加头数并不能改善GAN训练。对于32×32分辨率，我们使用补丁大小4×4，生成64个补丁的序列长度。对于64×64分辨率，我们只需将补丁大小增加到8×8，并保持与32×32分辨率中的序列长度相同。平移，颜色，切割，缩放数据扩展[59,25]的应用概率为0.8。所有基于基线变压器的GAN模型，包括我们的，使用平衡一致性正则化(bCR)， $λ_{real}=λ_{fake}= 10.0$。除了bCR，我们不使用正则化方法通常用于训练vit[51]，如Dropout, weight decay，或Stochastic Depth。我们发现，LeCam正则化[53]，类似于bCR，提高了性能。但为了更清晰的消融，我们不包括LeCam正则化。我们用Adam来训练我们的模型，$β1= 0.0， β2= 0.99$，[27]练习后的学习率为0.002。此外，我们采用非饱和logistic损失[18]，指数滑动平均[24]和均衡学习率[24]。我们使用128个小批量。ViTGAN和StyleGAN2都是基于Tensorflow 2实现[36]。我们在谷歌云TPU v2-32和v3-8上训练我们的模型。

###  Main Results

![image-20211212160940538](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/12/160943-984006.png)

### Ablation Studies

![image-20211212161113820](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/12/161114-817972.png)

![image-20211212161146063](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/12/161146-870490.png)

## More Quantitative Results

![image-20211212161308284](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/12/161308-846160.png)

## Implementation Notes

### Patch Extraction

我们使用了一个简单的技巧，通过允许图像补丁之间的一些重叠来减轻基于维特的鉴别器的过拟合。对于patch的每个边界边缘，我们将其扩展为o像素，使有效的patch大小(P+ 2o)×(P+ 2o)，其中o= p/2。虽然这个操作与一个具有核(P+ 2o)×(P+ 2o)和strideP×P的卷积操作有联系，但由于在我们的实现中没有使用卷积，所以在我们的模型中并没有将其视为卷积算子。注意，V anilla ViT[15]中的(非重叠)补丁的提取也与kernelP×Pand strideP×P的卷积操作有连接。

### Positional Embedding

ViT网络的每一个位置嵌入都是一个贴片位置的线性投影，后面跟着一个正弦激活函数。patch的位置被标准化到−1.0和1.0之间。

### Implicit Neural Representation for Patch Generation

每个位置嵌入都是像素坐标的线性投影，后面跟着一个正弦激活函数(因此称为傅里叶编码)。${P^2}$像素的像素坐标被标准化到−1.0和1.0之间。2层MLP将位置嵌入$E_{fou}$作为其输入，并以[27,1]中通过权重调制的patch嵌入为条件。
