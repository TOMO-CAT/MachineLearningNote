# Adaboost算法和提升树

## 前言
前面我们已经通过数学公式验证过，将众多“好而不同”的弱学习器组合起来可以提升模型的准确性。并且根据个体学习器之间是否存在强依赖关系，**我们将集成学习分为`boosting`和`bagging`两大类**（强依赖性体现在弱学习器的抽样方法）。

> 本篇我们主要讲`boosting`算法中的代表性提升算法`AdaBoost`，然后介绍提升方法的实例——提升树`boosting tree`

## Adaboost算法原理
#### 步骤一
假设每个训练样本在基本分类器中的作用相同，从而在等权重的原始数据上学习第一个基分类器$$G_1(x)$$
#### 步骤二
在第$$m$$轮（$$m=1,2,3,...,M$$）上顺次执行以下操作：

- 使用当前的权重分布$$D_m$$学习基分类器$$G_m(x)$$
- 计算基分类器$$G_m(x)$$在加权训练数据集上的分类误差率：
$$
e_m = P(G_m(x \neq y_i)=\sum_{i=1}^{N}w_{mi}I(G_m(x_i)\neq y_i)
$$
> 可以看到第二个等号表示分类误差率等于被$$G_m(x)$$误分类样本的权值之和

- 计算$$G_m(x)$$的系数$$\alpha_m$$ 
当$$e_m\leq\frac{1}{2}$$时$$\alpha_m\geq0$$，并且$$\alpha_m$$随着$$e_m$$的减小而增大，这也意味着分类误差率越小的基本分类器在最终分类器中的作用越大
- 更新训练数据的权值分布为下一轮做准备：
$$
w_{m_1,i}=
\begin{cases}
\frac{w_{mi}}{Z_m}e^{-\alpha_m},G_m(x_i)=y_i \\
\frac{w_{mi}}{Z_m}e^{\alpha_m},G_m(x_i)\neq y_i
\end{cases}
$$
可以看到误分类样本的权值扩大，而被正确分类样本的权值却得以缩小。因此误分类样本在下一轮学习中会起更大的作用。
> 不改变所给的训练数据，但是不断改变训练数据权值的分布，使得训练数据在基本分类器的学习中起不同的作用，这就是`AdaBoost`的一个特点。

#### 步骤三
通过系数$$\alpha_m$$将多个基分类器组合起来加权表决预测的分类结果。
> 注意$$\alpha_m$$之和不为1，$$f(x)$$的符号决定实例$$x$$的类，$$f(x)$$的绝对值表示分类的确信度。

## AdaBoost算法实现
基本原理依然是从训练数据中学习出一系列的弱分类器，并将弱分类器组合成一个强分类器。 
**输入**：训练集$$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$$，其中$$y_i$$取值为$$\{-1,+1\}$$ 
**输出**：最终分类器$$G(x)$$  

1. 初始化训练数据的权值分布

$$
D_1=(w_11,...,w_1i,...,w_1N),w_{1i}=\frac{1}{N}
$$
2. 对于$$m=1,2,...,M$$  
- 使用具有权值分布的$$D_m$$的训练数据学习，得到基分类器  

$$
G_m(x):X \rightarrow \{-1,+1\}
$$

- 计算$$G_m(x)$$在训练数据集上的分类误差率  
$$
e_m=P(G_m(x)\neq y_i)=\sum_{i=1}^{N}w_{mi}I(G_m(x_i)\neq y_i)  
$$
- 计算$$G_m(x)$$的系数
$$
\alpha_m=\frac{1}{2}log\frac{1-e_m}{e_m}
$$
- 更新训练数据集的权值分布

$$
D_{m+1}=(w_{m+1,1},...,w_{m+1,i},...,w_{m+1,N})
$$
$$
w_{m+1,i}=\frac{w_{mi}}{Z_m}exp(-\alpha_my_iG_m(x_i))
$$
其中，$$Z_m$$是规范化因子，它使得$$D_{m+1}$$成为一个概率分布：
$$
Z_m = \sum_{i=1}^{N}w_{mi}exp(-\alpha_my_iG_m(x_i))
$$
3. 构建基分类器的线性组合：
$$
f(x)=\sum_{m=1}^{N}\alpha_mG_m(x)
$$
最终分类器表示为：
$$
G(x)=sign(f(x))=sign(\sum_{m=1}^{M}\alpha_mG_m(x))
$$
## 提升树（boosting tree）
#### 1. 提升树模型
提升树指采用加法模型（基函数的线性组合）与前向分布算法，同时以决策树为基函数的提升方法。对于分类问题而言是二叉分类树，但对于回归问题而言是二叉回归树。 
提升树模型可以表示为决策树的加法模型：
$$
f_M(x)=\sum_{m=1}^{M}T(x;\Theta_m)
$$
其中，$$T(x;\Theta_m)$$表示决策树，$$\Theta_m$$表示决策树的参数，$$M$$表示树的棵树。
#### 2. 提升树算法原理
首先确定初始提升树$$f_0(x)=0$$，然后第$$m$$步的模型是：
$$
f_m(x)=f_{m-1}(x)+T(x;\Theta_m)
$$
其中下一棵决策树的参数$$\Theta_m$$通过经验风险最小化确定：
$$
\hat{\Theta}_m=argmin\sum_{i=1}^{N}L(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m))
$$
#### 3. 提升树算法类型
当使用的损失函数不同时，便对应着不同类型的提升树算法
* 二分类提升树  

直接将AdaBoost算法中的基本分类器限制为二叉树即可
* 回归提升树  

树可以表示为：
$$
T(x;\Theta)=\sum_{j=1}^{J}c_jI(x\in R_j)
$$
其中我们将输入空间划分为$$J$$个互不相交的区域$$R_1,R_2,...R_J$$，并且在每个区域确定输出的常量$$c_j$
回归树算法的具体细节可以看：
> [深入浅出机器学习算法：决策树引论和CART算法](https://zhuanlan.zhihu.com/p/70776194)
