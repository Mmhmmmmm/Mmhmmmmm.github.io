## **[KG-BERT: BERT for Knowledge Graph Completion(2019)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.03193)**

这篇文章是介绍知识库补全方面的工作，结合预训练模型BERT可以将更丰富的上下文表示结合进模型中，在三元组分类、[链接预测](https://www.zhihu.com/search?q=链接预测&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"91052495"})以及关系预测等任务中达到了SOTA效果。

具体的做法也非常简单易懂，就是修改了BERT模型的输入使其适用于知识库三元组的形式。

![img](https://pic1.zhimg.com/v2-dd8cf5259b8ff15120b7908824353e28_b.jpg)首先是**KG-BERT(a)**，输入为三元组 $(h,r,t)$的形式，当然还有BERT自带的special tokens。举个栗子，对于三元组  $(SteveJobs,foouned,AppleInc)$，上图中的`Head Entity`输入可以表示为`Steven Paul Jobs was an American business magnate, entrepreneur and investor`或者`Steve Jobs`，而`Tail Entity`可以表示为`Apple Inc. is an American multinational technology company headquartered in Cupertino, California`或`Apple Inc`。也就是说，头尾实体的输入可以是**实体描述**句子或者**实体名**本身。

模型训练是首先分别构建**[positive triple set](https://www.zhihu.com/search?q=positive+triple+set&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"91052495"})**和**negative triple set**，然后用BERT的[CLS]标签做一个sigmoid打分以及最后交叉熵损失

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D%3D-%5Csum_%7B%5Ctau+%5Cin+%5Cmathbb%7BD%7D%2B%5Ccup+%5Cmathbb%7BD%7D%5E%7B-%7D%7D%5Cleft%28y_%7B%5Ctau%7D+%5Clog+%5Cleft%28s_%7B%5Ctau+0%7D%5Cright%29%2B%5Cleft%281-y_%7B%5Ctau%7D%5Cright%29+%5Clog+%5Cleft%28s_%7B%5Ctau+1%7D%5Cright%29%5Cright%29)

![img](https://pic3.zhimg.com/v2-fc24b1e3a98066bd914fe54af731f846_b.jpg)

上述的**KG-BERT(a)**需要输入关系，对于关系分类任务不适用，于是作者又提出一种**KG-BERT(b)**，如上图。这里只是把sigmoid的二分类改成了softmax的关系多分类。

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal%7BL%7D%5E%7B%5Cprime%7D%3D-%5Csum_%7B%5Ctau+%5Cin+%5Cmathbb%7BD%7D%5E%7B%2B%7D%7D+%5Csum_%7Bi%3D1%7D%5E%7BR%7D+y_%7B%5Ctau+i%7D%5E%7B%5Cprime%7D+%5Clog+%5Cleft%28s_%7B%5Ctau+i%7D%5E%7B%5Cprime%7D%5Cright%29)

## **[K-BERT: Enabling Language Representation with Knowledge Graph(2019)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.07606)**

作者指出通过公开语料训练的BERT模型仅仅是获得了general knowledge，就像是一个普通人，当面对特定领域的情境时（如医疗、金融等），往往表现不如意，即**domain discrepancy**。而本文提出的**K-BERT**则像是领域专家，通过将知识库中的结构化信息（三元组）融入到预训练模型中，可以更好地解决领域相关任务。如何将外部知识整合到模型中成了一个关键点，这一步通常存在两个难点：

- **Heterogeneous Embedding Space：** 即文本的单词embedding和知识库的实体实体embedding通常是通过不同方式获取的，使得他俩的向量空间不一致；
- **Knowledge Noise：** 即过多的知识融合可能会使原始句子偏离正确的本意，过犹不及。

模型的整体框架如下图，主要包括了四个子模块： **knowledge layer**, **embedding layer**, **seeing layer** 和 **mask-transformer**。对于一个给定的输入 $s={w_0,w_1,w_2,...,w_n}$，首先 **knowledge layer**会从一个KG中注入相关的三元组，将原来的句子转换成一个knowledge-rich的句子树；接着句子树被同时送入**embedding layer**和**seeing layer**生成一个token级别的embedding表示和一个可见矩阵（visible matrix)；最后通过**mask-transformer**层编码后用于下游任务的输出。

![img](https://pic4.zhimg.com/v2-bf6449f45518cf3f88af7a62affb0247_b.jpg)

### **Knowledge Layer**

这一层的输入是原始句子 $s=\{w_0,w_1,w_2,...,w_0\}$ ，输出是融入KG信息后的句子树$t=\{w_0,w_1,...,w_i\{(r_{i,0},w_{i,0},...,r_{i,k},w_{i,k})\},...w_n\}$

通过两步完成：

- **K-Query** 输入句子中涉及的所有实体都被选中，并查询它们在KG中对应的三元组 $E$ ；
- **K-Inject** 将查询到的三元组注入到句子$S$ 中，将  $E$中的三元组插入到它们相应的位置，并生成一个句子树  $t$。

### **Embedding Layer**

K-BERT的输入和原始BERT的输入形式是一样的，都需要token embedding, position embedding和segment embedding，不同的是，K-BERT的输入是一个句子树，因此问题就变成了句子树到序列化句子的转化，并同时保留结构化信息。

![img](https://pic4.zhimg.com/v2-d1a6c85faf6a3db43125f56e96ac672b_b.jpg)**Token embedding**

句子树的序列化，作者提出一种简单的重排策略：**分支中的token被插入到相应节点之后，而后续的token被向后移动**。举个栗子，对于上图中的句子树，则重排后变成了`Tim Cook CEO Apple is visiting Beijing capital China is a City now`。没错，这的确看上去毫无逻辑，但是还好后面可以通过trick来解决。

### **Soft-position embedding**

通过重排后的句子显然是毫无意义的，这里利用了position embedding来还原回结构信息。还是以上图为例，重排后，`CEO`和`Apple`被插入在了`Cook`和`is`之间，但是`is`应该是接在`Cook`之后一个位置的，那么我们直接把`is`的position number 设置为3即可。Segment embedding 部分同BERT一样。

### **Seeing Layer**

作者认为Seeing layer的mask matrix是K-BERT有效的关键，主要解决了前面提到的**Knowledge Noise**问题。栗子中`China`仅仅修饰的是`Beijing`，和`Apple`半毛钱关系没有，因此像这种token之间就不应该有相互影响。为此定义一个可见矩阵，判断句子中的单词之间是否彼此影响

![[公式]](https://www.zhihu.com/equation?tex=M_%7Bi+j%7D%3D%5Cleft%5C%7B%5Cbegin%7Barray%7D%7Bcc%7D%7B0%7D%26%7Bw_%7Bi%7D+%5Cominus+w_%7Bj%7D%7D+%5C%5C+%7B-%5Cinfty%7D+%26+%7Bw_%7Bi%7D+%5Coslash+w_%7Bj%7D%7D%5Cend%7Barray%7D%5Cright.)

### **Mask-Transformer**

BERT中的Transformer Encoder不能接受上述可见矩阵作为输入，因此需要稍作改进。Mask-Transformer是一层层[mask-self-attention](https://www.zhihu.com/search?q=mask-self-attention&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"91052495"})的堆叠，

![[公式]](https://www.zhihu.com/equation?tex=+%5Cbegin%7Barray%7D+%7B+c+%7D+%7B+Q+%5E+%7B+i+%2B+1+%7D+%2C+K+%5E+%7B+i+%2B+1+%7D+%2C+V+%5E+%7B+i+%2B+1+%7D+%3D+h+%5E+%7B+i+%7D+W+_+%7B+q+%7D+%2C+h+%5E+%7B+i+%7D+W+_+%7B+k+%7D+%2C+h+%5E+%7B+i+%7D+W+_+%7B+v+%7D+%7D+%5C%5C+%7B+S+%5E+%7B+i+%2B+1+%7D+%3D+%5Coperatorname+%7B+softmax+%7D+%5Cleft%28+%5Cfrac+%7B+Q+%5E+%7B+i+%2B+1+%7D+K+%5E+%7B+i+%2B+1+%7D+%2B+M+%7D+%7B+%5Csqrt+%7B+d+_+%7B+k+%7D+%7D+%7D+%5Cright%29+%7D+%5C%5C+%7B+h+%5E+%7B+i+%2B+1+%7D+%3D+S+%5E+%7B+i+%2B+1+%7D+V+%5E+%7B+i+%2B+1+%7D+%7D+%5Cend%7Barray%7D+)

![img](https://pic4.zhimg.com/v2-4689a5cd50175ec3a36e901b8bb6b733_b.jpg)