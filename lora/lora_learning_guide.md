# lora
## 1.lora(Low-Rank Adaptation)概述
### 前景：过参数化模型的实际行为可以通过低维结构有效逼近。
    过参数化模型的实际行为可以通过低维结构有效逼近。LoRA方法提出后，极大推动了大模型参数高效微调（PEFT）的发展。它冻结了预训练的模型权重，并在 Transformer 架构的每一层注入可训练的秩分解矩阵，从而大大减少了下游任务的可训练参数数量。
## 2.原理
**a:基本思路** 

- 对于预训练的权重矩阵  $W_0 \in \mathbb{R}^{d \times k}$，在在训练期间，$W_0$被冻结并且不进行梯度更新。

- 令 $\Delta W = B \cdot A$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$。训练仅更新$A,B$。

**b:前向传播公式** 

- 新的权重为 $W = W_0 + \frac{\alpha}{r} B A$。
- 输入$x$，输出为 $y = (W_0 + \frac{\alpha}{r} B A) x$。

**c:参数量计算** 

- 原始更新参数：$d \times k$

- LoRA更新参数：$d \times r + r \times k = r \times (d+k)$

- $r$越小，省参数越多。

**d:缩放因子 $\alpha$**

- 控制低秩增量矩阵对整体的影响，通常$\alpha = r$，但可按实际调整。

>**$r ≪ min(d, k)$**
>
>相较于更新整个权重矩阵也即$d×k$个数据，在lora中只需要更新$d×r+r×k=r×(d+k)$,由于$r ≪ min(d, k)$，故$需要更新的参数<<权重矩阵的数量$
>
>**缩放因子:α**
>
>实际缩放为$α/r$
>控制低秩矩阵对原始矩阵的影响

## 3.优点
1. **参数效率高**  
   LoRA 仅训练低秩矩阵，大幅减少可训练参数数量，节省内存和计算资源。

2. **训练速度快、资源消耗低**  
   因为参数少，训练更快，更适合在低算力设备上进行微调。

3. **保持原始模型不变，便于部署与多任务扩展**  
   原始模型权重冻结，保证稳定性；不同任务可使用不同 LoRA 模块，便于复用与部署。

### 4.与其他PEFT方法对比

| 方法          | 训练参数量 | 适用模型      | 部署兼容性 | 参数隔离性 |
| ------------- | ---------- | ------------- | ---------- | ---------- |
| LoRA          | 极低       | Transformer等 | 高         | 强         |
| Adapter       | 低         | 广泛          | 高         | 强         |
| Prompt-tuning | 很低       | 需支持        | 中         | 较强       |
| Prefix-tuning | 很低       | 需支持        | 中         | 较强       |

### 5. LoRA 的最新变体与改进

- **QLoRA**: 结合量化与LoRA，极致节省内存（《QLoRA: Efficient Finetuning of Quantized LLMs》）。

- **AdaLoRA**: 自适应选择秩 r，使LoRA训练参数自适应分配（《AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning》）。

- **LoRA+**: 针对多层级LoRA设计，更高效表达（《LoRA+: Efficient Low Rank Adaptation of Large Models》）。

**参考资料**：[原论文](https://arxiv.org/abs/2106.09685)
[视频讲解](https://www.bilibili.com/video/BV1r9ieYhEuZ?spm_id_from=333.788.videopod.episodes&vd_source=98b13e790f651b8b3a1a8a1b889d1f15&p=11)
[相关博客](https://spaces.ac.cn/archives/9590)
