# R-Drop

## 1. 数学原理与损失函数

设模型输出为 softmax 概率分布：

- 输入 $x$，输出 $p_1 = f(x;\theta_1)$ 和 $p_2 = f(x;\theta_2)$，其中 $\theta_1$, $\theta_2$ 是 Dropout 下的参数子集

定义损失函数如下：

$
\mathcal{L} = \text{CE}(p_1, y) + \text{CE}(p_2, y) + \alpha \cdot \left[\text{KL}(p_1 \parallel p_2) + \text{KL}(p_2 \parallel p_1)\right]
$

- $\text{CE}(p, y)$ 是标准交叉熵
- $\text{KL}(p_1 \parallel p_2)$ 是 Kullback-Leibler 散度
- $\alpha$ 是控制 KL 项影响力的超参数（通常取 1~5）
>*主要思想*：已知dropout中p参数固定，但是被drop掉的神经元不固定，因此可能导致相同输入的情况下，输出结果不同，为了解决这个问题引入了kl散度，本来kl散度的作用也是尽量让两个分布概率拟合，所以在损失函数中ce还是占主导，其主要目的还是监督模型向正确输出靠拢，而kl只是让模型对dropout不那么敏感，从而提高模型的鲁棒性。
>

---
  
## KL散度

## 什么是 KL 散度？

KL 散度（Kullback-Leibler Divergence）是用来衡量两个概率分布之间差异的非对称指标。

简而言之：**衡量“一个分布距离另一个分布有多远”**。
##  数学定义

设有两个概率分布 $P$ 与 $Q$（通常 $P$ 是“真实分布”，$Q$ 是“预测分布”）：

### (有两种形式这里是)离散形式：

$D_{\mathrm{KL}}(P \parallel Q) = \sum_{i} P(i) \log \left( \frac{P(i)}{Q(i)} \right)$

---

| 对比项       | 正向 KL（KL(P‖Q)）        | 逆向 KL（KL(Q‖P)）        |
|--------------|---------------------------|---------------------------|
| 优化趋势     | 覆盖目标分布所有区域       | 聚焦于目标的高概率区域     |
| 行为偏好     | 保守，防止遗漏             | 激进，可能模式崩溃         |
| 常见用途     | 变分推理、R-Drop           | 强化学习、模型压缩         |

[通俗易懂解释]("https://www.youtube.com/watch?v=q0AkK8aYbLY")