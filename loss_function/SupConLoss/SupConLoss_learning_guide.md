# SupConLoss
## 1.概述
在对比学习的基础上引入 **标签** 信息，将同一类别的样本都视为正样本对，从而增强监督信号。

例如：

- SimCLR：对同一张图片进行增强（如 A 和 B），二者为正样本对；
- SupCon：所有属于同一类别（如“猫”）的图片之间（包括不同原始图片）都是正样本对。

主要目的：让同类样本特征更加聚集，不同类样本特征更分离。

* 损失函数
    $$
    \mathcal{L}^{\text{sup}}_{\text{out}} = \sum_{i \in I} \mathcal{L}^{\text{sup}}_{\text{out}, i} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}
    $$

* 参数说明（假设共有 N 个样本，每个样本生成两个增强样本，增强样本数量又名 views）：

    - $$i \in I \equiv \{1, \dots, 2N\}$$：选定的锚点索引；
    - $$A(i) \equiv I \setminus \{i\}$$：去除自身后的样本索引集合；
    - $$(z_\ell = \text{Proj}(\text{Enc}(\tilde{x}_\ell)) \in \mathbb{R}^{D_P}$$：样本的向量表示；
    - $$P(i) \equiv \{p \in A(i) : \tilde{y}_p = \tilde{y}_i\}$$：和锚点属于同一类别的样本索引集合；
    - $$|P(i)|$$：与锚点同类样本的数量；
    - $$\tau$$：温度参数。

## 2.代码解读
### 掩码生成
features = [batch_size,views,features_dim]  
mask = [batch_size,batch_size]
```
if labels is not None and mask is not None:
	raise ValueError('Cannot define both `labels` and `mask`')
elif labels is None and mask is None:
	mask = torch.eye(batch_size, dtype=torch.float32).to(device)
elif labels is not None:
	labels = labels.contiguous().view(-1, 1)
	if labels.shape[0] != batch_size:
		raise ValueError('Num of labels does not match num of features')
		mask = torch.eq(labels, labels.T).float().to(device)
else:
	mask = mask.float().to(device)
```
**掩码矩阵生成逻辑：**

- `labels` 和 `mask` 不可同时提供；
- 若二者都不提供，退化为 SimCLR；
- 若提供 `labels`，则通过 `torch.eq` 生成掩码，表示两个样本是否同类（1 为同类，0 为异类）。

---
### 构建锚点与对比样本
contrast_count = views  
contrast_feature = [views×batch_size,features_dim]//每一行就是一个样本的向量  
```
 contrast_count = features.shape[1]
 contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
 if self.contrast_mode == 'one':
 	anchor_feature = features[:, 0]
    anchor_count = 1
 elif self.contrast_mode == 'all':
 	anchor_feature = contrast_feature
    	anchor_count = contrast_count
 else:
 	raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
```
- `contrast_count`：增强视图数量；
- `contrast_feature`：将不同视图拼接成一个大的特征矩阵；
- `anchor_feature`：锚点（可以是一个视图或所有视图）；
- `torch.unbind`：按视图拆分。

---
### 计算相似度 & 数值稳定
anchor_dot_contrast = [batch_size,views×batch_size]  
logits = [batch_size,views×batch_size]
```
 # compute logits
 anchor_dot_contrast = torch.div(
    	torch.matmul(anchor_feature, contrast_feature.T),
        self.temperature)
 # for numerical stability
 logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
 logits = anchor_dot_contrast - logits_max.detach()
```
- 计算锚点与其他所有样本的余弦相似度（dot product）；
- 使用 `logits_max` 做数值稳定（避免 softmax 中出现指数爆炸）。

---
### 屏蔽自身对比 & 掩码扩展
mask = [batch_size×1,batch_size×views]  
logits_mask = [batch_size×1,batch_size×views]//除了对角线为0其余全为1
```
# tile mask
mask = mask.repeat(anchor_count, contrast_count)
# mask-out self-contrast cases
logits_mask = torch.scatter(
	torch.ones_like(mask),
	1,
	torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
	0
)
mask = mask * logits_mask
```
- [torch.repeat]("https://blog.csdn.net/flyingluohaipeng/article/details/125039368")，最后得到的mask矩阵，是从源mask矩阵中，除去每个样本不应该与自己进行对比学习

- 使用 `repeat` 拓展掩码矩阵到多锚点场景；
- `logits_mask`：将对角线置为 0，防止样本与自身对比。

---
### 计算对数概率
exp_logit = [batch_size,views×batch_size]  
log_prob = [batch_size,views×batch_size]
```
 # compute log_prob
 exp_logits = torch.exp(logits) * logits_mask
 log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
```
- `log_prob[i]` 表示第 i 个样本与所有其他样本之间的 log-softmax 相似度。
- 分母中为所有非自身样本的 softmax 和，体现对比概率归一化。

---
### 正样本期望 log 概率
mean_log_prob_pos = [batch_size,1]
```
 mask_pos_pairs = mask.sum(1)
 #防止除以0
 mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
 mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
```
- `mask_pos_pairs`：每个锚点的正样本数量；
- `torch.where` 防止除零错误；
- 最终计算出每个锚点的平均 log 概率（只统计正样本）。

---
## 计算损失

```
  loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
```
- 得到每个样本的损失值，并进行平均。

## 参考资料
[代码]("https://github.com/HobbitLong/SupContrast/blob/master/losses.py")  
[论文]("https://arxiv.org/abs/2004.11362")
