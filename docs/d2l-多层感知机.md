### 4.1 多层感知机
MLP(Multi-Layer Perception)多层感知机

比如说2个仿射变换，实际上可以归纳到1个仿射变换。那么就没有必要设置多层感知机了。
因而引入了激活函数 $\sigma$ 。有各种各样的激活函数，包括ReLU、sigmoid等等。

### 4.2 多层感知机的从零开始实现

**4.1 习题**
1. 计算pReLU激活函数的导数。
> $$pReLU = max(0,x) + a * min(0,x)$$
> $$x>0,\frac{dy}{dx} = 1$$
> $$x<0,\frac{dy}{dx} = a$$
2. 证明一个仅使用ReLU（或pReLU）的多层感知机构造了一个连续的分段线性函数。
> 考虑ReLU函数 $y = max(0,x)$
> $$ \lim_{x \to 0^-} = 0$$
> $$ \lim_{x \to 0^+} = 0$$
> $$ \therefore ReLU 是连续的 $$
3. 证明$tanh(x) + 1 = 2sigmoid(2x)$。
> $$ tanh(x) = \frac{1-exp(-2x)}{1+exp(-2x)} $$
> $$ sigmoid(x) = \frac{1}{1+exp(-x)} $$
> $$ tanh(x) + 1 = \frac{2}{1+exp(-2x)} == 2 \frac{1}{1+exp(-2x)} = 2sigmoid(2x)$$
4. 假设我们有一个非线性单元，将它一次应用于一个小批量的数据。这会导致什么样的问题？
> 非线性单元通常用于捕捉数据间的复杂线形关系，但如果数据量不足，模型可能无法学习到这些复杂的关系。