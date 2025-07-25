## LoRa代码实现(Linear层为例)
- 所有Lora层基类实现
```
class LoRALayer():
    def __init__(
        self, 
        r: int, #低秩矩阵的秩
        lora_alpha: int, #缩放因子
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Dropout 层（可选）
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # 是否在训练/评估时合并权重
        self.merge_weights = merge_weights
        self.merged = False
```
>lora在训练时是否合并权重，即(W = W₀ + BA|三者独立(LoRA矩阵单独存储和更新)**默认**。)
>训练阶段默认不合并
- Linear初始化
```
class Linear(nn.Linear, LoRALayer):
   def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
```
>主要实现：低秩矩阵lora_A,lora_B的创建，通过nn.parameter()将其设置为需要梯度更新，self.weight.requires_grad = False冻结原来的权重矩阵

- 前向传播
```
def forward(self, x: torch.Tensor):
    #判断是否转置
    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    if self.r > 0 and not self.merged:
        result = F.linear(x, T(self.weight), bias=self.bias)
        result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result
    else:
        return F.linear(x, T(self.weight), bias=self.bias)
```
>如果启用了 LoRA 且未合并权重，使用原始权重进行线性变换，再加上 LoRA 的增量项：(x @ A^T @ B^T) * scaling

- 训练/评估切换
```
def train(self, mode: bool = True):
    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    nn.Linear.train(self, mode)
    if mode:
        if self.merge_weights and self.merged:
            # 训练时取消合并
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    else:
        if self.merge_weights and not self.merged:
            # 推理时合并权重
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
```
>在训练模式下:如果已合并权重，则将其“减回去”，恢复原始权重.
>在评估模式下，将 LoRA 权重合并进原始权重中，提升推理速度.

参考资料：https://github.com/microsoft/LoRA/blob/main/loralib



