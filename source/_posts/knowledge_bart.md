# BART

BART是Bidirectional and Auto-Regressive Transformers的简写，来自论文：[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1910.13461.pdf)

## 背景：Seq2Seq预训练

去年10月，来自Google和Facebook的团队分别发布了新的Transformer-related论文：T5和BART。 这两篇论文在如抽象总结和对话等生成任务上都取得了更好的下游性能，主要有两个改变：

- 在BERT的双向编码器架构中添加因果解码器；
- 用更复杂的预训练任务代替BERT的完形填空任务。

### **Bert vs. GPT2**

正如BART作者在论文中写的，

> (BART) can be seen as generalizing Bert (due to the bidirectional encoder) and GPT2 (with the left to right decoder).

### **BERT**

ERT最重要的预训练任务是预测masked token，并使用整个输入来获取更完全的信息以进行更准确的预测。这对于那些允许利用位置$i$之后的信息来预测位置$i$的任务是有效的，但是对于诸如文本生成之类的任务则没有多大用处，这些对位置$i$的预测只能取决于先前生成的单词。

在BERT源码中，在预测位置$i$时可以使用哪些信息是由由一个称为`attention_mask`的参数来控制的， 注意掩码中的值为1表示模型在预测行的单词时可以利用的列单词的信息。

下图是BERT的"Fully-visible" 注意力矩阵，

![img](https://pic1.zhimg.com/80/v2-f964940fa1025254317a087cfb27bb0c_1440w.jpg)

- ![img](https://pic1.zhimg.com/80/v2-ac5de7890849432e16681a881fea5e50_1440w.png)

- 优点：本质为降噪自编码特征表示，通过引入噪声[MASK]构建MLM，获取上下文相关的双向特征表示；

- 引入独立性假设，为**联合概率的有偏估计**，没有考虑预测[MASK]之间的相关性

- - 不适合直接处理生成任务，MLM预训练目标的设置造成预训练过程和生成过程不一致；
  - 预训练时的[MASK]噪声在finetune阶段不会出现，造成两阶段不匹配问题；

- 代表模型：BERT系列模型；

### **GPT**

GPT预训练任务使用的是`autoregressive`的思想，使用已经解码出的信息来预测下一个位置。该种模式对于生成任务更为有效，而对于那些可以使用全局输入来得到输出的下游任务则比较差劲。

同样的，给出GPT的注意力矩阵，

![img](https://pic1.zhimg.com/80/v2-bdf9db30fdceb59a9d3d3a07c1a67300_1440w.jpg)

在这里，当我们预测`eating`时，可以使用的信息只有`<BOS> I love`。

![img](https://pic3.zhimg.com/80/v2-4f6492d8b4303d278441bbb4c933a20a_1440w.png)

- 优点：

- - 文本序列**联合概率的密度估计**，即为传统的语言模型，天然适合处理自然生成任务；

- 缺点：

- - 联合概率按照文本序列从左至右分解（**顺序拆解**），无法通过上下文信息进行双向特征表征；

- 代表模型：ELMO/GPT1.0/GPT2.0；

- 改进：XLNet将传统的自回归语言模型进行推广，将顺序拆解变为**随机拆解**（排列语言模型），产生上下文相关的双向特征表示；

### **Encoder-Decoder**

我们的新朋友，例如BART，可以做到两全其美。

其中Encoder的注意力矩阵是`Fully-visible`的，

![img](https://pic1.zhimg.com/80/v2-f964940fa1025254317a087cfb27bb0c_1440w.jpg)

而Decoder的注意力矩阵是`autoregressive`,

![img](https://pic3.zhimg.com/80/v2-c1c1051fa14cc3641c0c764f0cad5b62_1440w.jpg)

编码器和解码器通过`cross attention`连接，其中每个解码器层都对编码器输出的最终隐藏状态进行attention操作，这会使得模型生成与原始输入紧密相关的输出。

### transformer

![img](https://s1.ax1x.com/2020/04/25/JyCdy9.png)

## Pre-training

- **Token Masking**：Following BERT (Devlin et al., 2019), random tokens are sampled and replaced with` [MASK] `elements.
- **Sentence Permutation**：A document is divided into sentences based on full stops, and these sentences are shuffled in a random order.
- **Document Rotation**：A token is chosen uniformly at random, and the document is rotated so that it begins with that token. This task trains the model to identify the start of the document.
- **Token Deletion**：Random tokens are deleted from the input. In contrast to token masking, the model must decide which positions are missing inputs.
- **Text Infilling**：A number of text spans are sampled, with span lengths drawn from a Poisson distribution (λ=3). Each span is replaced with a single` [MASK] `token. 0-length spans correspond to the insertion of `[MASK]` tokens. Text infilling teaches the model to predict how many tokens are missing from a span.

![img](https://z3.ax1x.com/2021/04/06/c18o11.png#shadow)

## **Fine-tuning**

### Sequence Classification Tasks

序列分类任务中，编码器和解码器的输入相同，解码器 token 的最终隐藏状态被输入到多类别线性分类器中。BART 在解码器最后额外添加了一个 token，如下图所示，该 token 位置的输出可以被认为是该句子的 representation

![img](https://pic2.zhimg.com/80/v2-869ae8f65a9f1be753a51ee2f9011f31_1440w.jpg)在上述示例中，原始文档为`A B C D E`。在编码之前将文本`[C，D]`屏蔽掉，又在B之前插入一个额外的掩码，然后将损坏的文档`A _ B _ E`作为编码器的输入。解码器必须使用编码器的输出和先前未损坏的标记来重建原始文档。

### Sequence Generation Tasks

由于 BART 具备自回归解码器，因此它可以针对序列生成任务进行直接微调，如问答或者文本摘要

### Machine Translation

作者采用新的随机初始化 Encoder 替换 BART 编码器的 `Embedding `层。该模型以端到端的方式进行训练，即训练一个新的编码器将外来词映射到输入。新的编码器可以使用不同于原始 BART 模型的词汇。其中随机初始化 Encoder 的训练分两步，均需要将来自 BART 模型输出的交叉熵损失进行反向传播。第一步，作者冻结 BART 的大部分参数，仅更新随机初始化的 Encoder、BART 位置嵌入和 BART 编码器第一层的自注意力输入投影矩阵。第二步，作者将所有模型参数进行少量迭代训练

![img](https://z3.ax1x.com/2021/04/06/c1GPnf.png)





## **Results**

![img](https://z3.ax1x.com/2021/04/06/c1GFHS.png)

从上表可以看出，貌似带上 Document Rotation 或 Sentence Shuffling 效果都不是太好，可以这么理解，假如模型在训练的时候看到的句子顺序都是乱的，它可能就认为这个世界的句子顺序都是乱的，当你做测试的时候，输入的句子是正序的，可能模型就不知所措了。实际上 Text Infilling 可以看作是 Token Masking+Token Deletion，所以 Text Infilling 效果这么好也可以理解

![img](https://z3.ax1x.com/2021/04/06/c1Guj0.png)

## 应用

## KM-BART: Knowledge Enhanced Multimodal BART for Visual Commonsense Generation

![image-20211223202316964](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/23/203112-858339.png)[2101.00419.pdf (arxiv.org)](https://arxiv.org/pdf/2101.00419.pdf)

![image-20211223202500339](https://gitee.com/ZhaoJW11/typroimg/raw/master/image/202112/23/202524-103198.png)

可以看到，整个模型的backbone就是一个BART，为了能适配多模态输入，论文通过在原始的BART中增加特殊的tokens来适应不同的预训练任务。在编码器部分，对于不同的预训练任务，通过加入几个特殊tokens来区分不同模态的输入：

- 常识生成：`<before>`, `<after>`和`<intent>`
- 属性和关系预测：`<region_caption>`
- 掩码预训练语言模型：`<caption>`

对于不同模态的输入，引入几个特殊tokens：

- 视觉embedding：`<img>`, `</img>`
- 事件文本：`<event>`, `</event>`
- 语言模型：`<mlm>`, `</mlm>`

解码器利用类似于GPT的单向Transformer，模型需要预测被masked的单词和被masked视觉区域的类别分布。

这项研究中到底哪里用了知识图谱？其实就在于其中一个预训练任务基于知识的常识生成任务，这个预训练任务使用了COMET，这是一个在常识知识图谱上得到的预训练语言模型，所以可以看到这篇论文所谓的视觉加知识图谱其实就是在知识融入的PLM中调整输入tokens保证可以编码视觉输入数据。

### 