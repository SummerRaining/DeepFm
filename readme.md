

__Fm流程__：what：factorization machine 因子分解机，用于获取稀疏特征的交互信息。

Why：对于高基的离散变量一般使用onehot的方式编码，而对特征的进行二阶交互会更大的增加特征数量，特征数量从n维到n(n-1)维。因此产生大量的稀疏特征。

How：FM是在逻辑回归上增加了二阶特征，通过对每个特征学习一个隐向量vi，任意两个特征的交互表示为<vi,vj>xixj。输出结果为：“线性部分”加上“所有二阶交互特征”，然后加上sigmoid激活函数。
$$
y_{fm} = w_0+<w,x>+\sum_{i,j\in(1,n)}<v_i,v_j>x_ix_j \\
output = sigmoid(y_{fm})
$$
优点分析：

1. FM的学习参数量为nd，d为隐向量的维数。相比于所有的二阶交互参数量为n(n-1）,减少很多。

2. 可以学习到训练集上不存在的交互特征，传统的二阶特征，如果训练集上的xixj全部为0，那么xixj的参数wij无法学习。而FM，可以通过<vi,vj>来计算得到。每个参数学习的数据量增加了。
3. Fm时间可以分解成，先相加再相乘- 先相乘再相加形式。时间复杂度为，O(nd)。

 

__Deepfm__：why:解决FM只能学习到二阶的特征，无法获得更高阶特征的缺点，在FM之外增加了dnn部分。

how：1. 线性部分和二阶Fm部分与FM一致。

2. 将FM学习的隐向量拼接在一起，添加多个全连接层，最后全连接到1维输出。
3. 将线性输出，二阶fm输出，dnn输出相加。使用sigmoid激活函数得到模型输出。

$$
y_{deepfm} = y_{fm}+y_{dnn} \\
y_{dnn} = dense(concate([v_1,...,v_n]))
$$



 

__NFM__：解决deepfm学习高阶交叉特征困难的问题，提出了bi-interaction pooling层。

how :1. 线性部分和fm一致。

2. 所有的特征学习隐层向量vi，令zi=xivi。
3. 将任意两个特征进行对应元素相乘zizj，所有交互特征共1/2n(n-1)个相加得到bi-interaction pooling的输出。
4. 之后接上多个全连接层，最后一层全连接到1维的输出。
5. 线性部分+dnn输出，使用sigmoid激活函数得到模型的输出。

优点：1、dnn的底层部分进行了特征的交互，使交互特征更容易学习。

2. bi-interaction pooling层计算量比较小O(nk)。

 

__Auto-Int__：why:认为交互特征的重要性不一致，使用multi-heads self-attention的方式动态的分配，交互特征的权重。

how: 1.将所有特征（离散和连续），都学习一个隐向量vi，令zi=xivi。

2. n个特征的zi拼接在一起得到矩阵W，接上multi-heads self-attention层。
3. 具体来讲，使用参数矩阵Q,K,V，与输入W相乘得到，向量qeury,key,value。
4. query i对每个key进行矩阵乘法得到一个分数，对n个分数进行softmax得到归一化的权重，权重对所有的value加权平均，得到的就是向量i的输出。
5. 将self-attention的输出和输入W相加，借鉴resnet的想法，防止模型退化。
6. 将该模型多次堆叠，然后使用全连接到1维向量，sigmoid激活函数。