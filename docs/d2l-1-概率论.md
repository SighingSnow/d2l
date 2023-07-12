d2l-概率论

Multinomial distribution 多项分布

Sample 抽样

为了抽取一个样本，即掷骰子，只需要传入一个概率向量，输出是另一个相同长度的向量：在索引i处的值是采样结果中i出现的次数。

```python
from torch.distribution import multinomial
from d2l import torch as d2l
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1,fair_probs).sample((500,)) # 表示每次实验sample 1次，共做500组实验
```

---

**Bayes' therom**
$$
P(A,B) = P(A|B)P(B)
$$

---

**独立性与相关性**：

独立性：$P(A,B) = P(A) * P(B)$

相关性：指的是线性相关程度。不相关，即线性无关。

---

**马尔可夫链（Markov Chains)**

考虑一个随机变量的序列$X = \{X_0,X_1,..., X_t, ...\} $，$X_t$表示时刻t的随机变量。假设在时刻0的随机变量$X_0$遵循分布$P(X_0) = \pi_0$称为初始状态，$t \ge 1,\exists P(X_t|X_{t-1}) $,如果$X_{t}$只依赖$X_{t-1}$，而不依赖于过去的随机变量$\{X_0,X_1,...,X_t-2\}$成为马尔可夫链

随机变量A，B，C，B只依赖A，C只依赖B，能简化联合概率$P(A,B,C)$吗。

如下为错误解答

> P(A,B,C) = P(A) + P(B) + P(C) - P(AB) - P(BC) - P(AC) + P(ABC)
>
> ​			  = P(A) + P(B) + P(C) - P(A) - P(B) - P(A) + P(A) =  P(C)

$$
P(A,B,C) = P(C|A,B) * P(A,B) = P(C|A,B) * P(A) * P(B|A) = P(C|B) * P(A) * P(B|A)
$$



![image-20230606171052129](/Users/songtingyu/Library/Application Support/typora-user-images/image-20230606171052129.png)

D is diagnosed, H is actually infected.

条件独立
$$
P(D_1,D_2 | H = 1) = P(D_1|H=1) * P(D_2|H=1)
$$
$P(D_1|H=1)$ 以及 $P(D_2|H=1)$

D1是第一次测试结果，D2是第二次测试结果

$P(D_1,D_2|H=1) = P(D_1|H=1) P(D_2|H=1)$

$P(D_1=D_2=1 | H=0) = 0.02 \neq P(D1=1|H=0) * P(D_2=1|H=0) = 0.1*0.001 = 0.0001 $

Given H=1,we have,

|      | 0        | 1             |
| ---- | -------- | ------------- |
| 0    | 0*0.02=0 | 1*0.98=0.98   |
| 1    | 0        | 1*0.02 = 0.02 |

已知条件

![image-20230607161143416](/Users/songtingyu/Library/Application Support/typora-user-images/image-20230607161143416.png)

![image-20230607161155895](/Users/songtingyu/Library/Application Support/typora-user-images/image-20230607161155895.png)

现在问$P(D_1,D_2|H=0)$的联合分布律。Given H=0

|      | D1=0 | D1=1 |
| ---- | ---- | ---- |
| D2=0 |      |      |
| D2=1 |      | 0.02 |

$\because P(D_1=1,D_2=0|H=0)+P(D_1=1,D_2=1|H=0) = P(D_1=1|H=0) = 0.01 $

$P(D_1=1,D_2=0|H=0) = 0.01-0.02 = ?$









