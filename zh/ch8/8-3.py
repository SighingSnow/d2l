import random
import torch
from d2l import torch as d2l

import sys
sys.path.append("../..")
from dl import torch_zh as dl

tokens = dl.tokenize(dl.read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = dl.Vocab(corpus)
print(vocab.token_freqs[:10])

freqs = [freq for token,freq in vocab.token_freqs]
# d2l.plot(freqs,xlabel = 'token: x', ylabel='frequency: n(x)',
#          xscale='log',yscale='log')
# d2l.plt.show()

# 从copus生成一个大小为batch_size的样本
# @param batch_size: 每个子序列 样本的数量
# @param num_steps : 每个子序列中预定义的时间步数
def seq_data_iter_random(corpus, batch_size, num_steps):
    # 取随机偏移量
    corpus = corpus[random.randint(0,num_steps-1):]
    # 子序列数量
    num_subseqs = (len(corpus)-1) // num_steps
    # len(inital_indices) = num_steps
    initial_indices = list(range(0,num_steps*num_subseqs,num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos+num_steps]
    
    # 每次挑选batch_size个子序列
    num_batches = num_subseqs // batch_size
    for i in range(0,batch_size*num_batches,batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 顺序分区：保证连续性
def seq_data_iter_sequential(corpus,batch_size,num_steps):
    # 这里的offset指的是从第几个token开始
    offset = random.randint(0,num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset+num_tokens])
    Ys = torch.tensor(corpus[offset+1:offset+num_tokens+1])
    Xs,Ys = Xs.reshape(batch_size,-1), Ys.reshape(batch_size,-1)
    # Xs.shape[1] = num_tokens / batch_size
    num_batchs = Xs.shape[1] // num_steps
    for i in range(0,num_batchs * num_steps,num_steps):
        X = Xs[:,i:i+num_steps]
        Y = Ys[:,i:i+num_steps]
        yield X,Y

class SeqDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        # self.corpus, self.vocab = d2l.l
        self.batch_size, self.num_steps = batch_size,num_steps
    
    def __iter__(self):
        return self.data_iter_fn(self.corpus,self.batch_size,self.num_steps)


myseq = list(range(35))
for X,Y in dl.seq_data_iter_random(myseq,batch_size=2,num_steps=5):
    print('X: ',X,'\nY: ',Y)

for X,Y in seq_data_iter_sequential(myseq,batch_size=2,num_steps=5):
    print('X:',X,'\nY:',Y)