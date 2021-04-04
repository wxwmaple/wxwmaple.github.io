---
layout: post
title: "如何预训练一个词向量"
subtitle: ''
author: "wxwmaple"
mathjax: true
header-img: "img/home-bg-used.jpg"
tags:
  - NLP
  - Embedding
  - 自然语言处理
  - 词嵌入
---
## 前言

词嵌入（word embedding）的本质是用向量来描述词的信息，主要的好处是能更清晰地刻画词的性质。

举个简单的例子，计算机可能并不知道pink是什么，但RGB代码(255, 192, 203)可以用三个维度刻画这种颜色的程度，十分清晰。当然，词嵌入未必是可解释的，比如一根64维的向量，你很难说清楚每一个维度究竟在描述什么。

## 词向量技术

语境无关：word2vec、GloVe、fastText

语境相关：XLNet、BERT

所谓语境无关，指的是某个特定的词只能映射到唯一的编码上去。这自然会产生很多问题，比如川普可能代表美国第45届总统，也有可能是四川普通话。如果都用一根Embedding Vector去表示，可能会影响效果。

### word2vec

word2vec技术的本质是借用神经网络语言模型来训练一个矩阵。

什么是语言模型？语言模型旨在建立一个概率分布$P(x_1,x_2,...,x_n)$，然后用可以用这个概率分布来衡量每一个句子序列$x=[x_1,x_2,...,x_n]$是不是人话。如果写得通俗一点的话，语言模型即$y=f(x)$，其中$x$是句子序列输入，$y$是人话指数：越接近1，$x$序列越像人话，越接近0则越不像。

那什么是神经网络语言模型呢？它自然是用神经网络的思想，训练上述的“人话鉴定器”啦。具体来说，它主要分为三个步骤：

1. 将词汇表中的每个词表示成一个在m维空间里的实数形式的分布式特征向量
2. 使用序列中词语的分布式特征向量来表示连接概率函数
3. 同时学习特征向量和概率函数的参数

有意思了！你是通过“将词表示成词向量”的方式去构造语言模型的？虽然你的思路很美，但马上就是我的了！word2vec借用了神经网络语言模型的外壳，表面上在训练一个语言模型，其实并不关心语言模型$f$函数本身，而是抽出了中间的副产物——一个矩阵作为我们想要的东西。这个矩阵其实隐含在上述的第一步中，他是一个二维的，第一维大小为词表总数，第二维大小为我们所需的映射维度（上文为m维）的一个矩阵。我们在输入阶段，本质输入的是词的one-hot向量（假设第p维是1吧），很容易注意到这样的向量去乘别的矩阵，刚好拿到的就是别的矩阵的第p行。好家伙，这个矩阵就叫Embedding矩阵吧，任意给个词id，先转one-hot，再和你一乘，就能生成Embedding Vector啦，真方便。

虽然语言模型是判断一整句话是不是人话，但它显然可以用来实现很多子任务。通过改变子任务，我们可以训练出不同种类的各具特色的词向量。

#### CBOW

任务：给出上下文步长各c个词，要求预测中心词。如c=2时，["毛","利","X","五","郎"]。

#### Skip-gram

任务：给出中心词，预测它的上下文。 如c=2时，["X","X","小","X","X"]。

了解到这里就足够了，训练过程比较复杂。我们实际上有开源的gensim包可以用。

```python
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

raw_sentences = ["I love caixukun .","I hate wuyifan ."]
sentences= [s.split() for s in raw_sentences]


model = word2vec.Word2Vec(sentences, min_count=1,vector_size=8)#训练8维的向量，且由于训练集太小，要求出现1次的词语都不能丢掉
model.wv["caixukun"]
#Out: array([ 0.11922649, -0.09148958, -0.02917212, -0.02422178,  0.10096794,
       -0.0741362 ,  0.00056452, -0.05942169], dtype=float32)
```



## 在深度学习框架中的Embedding

在很多框架中（如keras/torch）都自带embedding层，但从某种意义上说，Embedding层和Dense层没有实质上的区别，因为他们的本质都是$y=Ax$。如果直接把embedding作为实际任务模型的一部分，它的参数固然会被更新，但此时的Embedding Vector就没有各种漂亮的特征了（比如向量的加减法）。

一般建议先用各种方法（gensim自己训练，或者下别人训练的参数）获取独立的Embedding矩阵，再接入Embedding层，并冻结参数。此时的Embedding层就和一般的Dense相比有点优势了：它是通过查表的方式直接拿向量，要比Dense的矩阵乘法操作快很多。

```python
#初始化embedding矩阵
embedding_matrix = np.zeros(word_index,the_dimension_ud_like_to_use)

#从外部导入embedding矩阵
#try to import your pretrained embedding matrix

#导入Embedding层并冻结训练
from keras.layers import Embedding
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
```

