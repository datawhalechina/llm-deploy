## 1.3：经典的量化方法（How-part2:QAT）
第三章讲经典QAT的LLM-QAT，OQ（涵盖weight-only 和 weight-act）-方式：原理讲解+代码，

### 1.3.1 LLM-QAT

#### 1）原理讲解

LLM-QAT（Large Language Model Quantization-Aware Training）是一种针对大型语言模型的量化感知训练方法。在LLM-QAT中，模型在训练过程中就考虑到了量化操作，从而可以减小量化后的模型性能损失。

LLM-QAT的基本思想是使用预训练模型自己生成的数据进行知识蒸馏，并在量化权重和激活的同时，对KV cache进行量化。以下是LLM-QAT的详细步骤：

1. **数据生成**：

   - 使用预训练模型生成数据。具体地，从词汇表中随机化第一个Token（例如<start>），并让预训练模型生成下一个Token。然后将生成的Token附加到起始Token以生成新的输出，重复这个迭代过程，直到达到句子Token的结尾或最大生成长度。
   - 为了生成更加多样化的句子，使用预训练模型的SoftMax输出作为概率，从分布中随机采样下一个Token。为了提高微调学生模型的准确性，采用混合采样策略，针对前3-5个Token确定性地选择top-1预测，然后剩余的Token进行随机采样。

2. **量化操作**：

   - 量化是指将连续的无限值映射到较小的离散有限值集合的过程。在LLM-QAT中，量化包括权重、激活和KV cache的量化。
   - 针对权重采用的是per-channel量化，而针对激活和KV cache采用的是per-token量化。
   - 在量化过程中，采用的是均匀线性对称量化，量化方法采用minmax方法（而不是lsq等方法）。采用对称量化的原因是观察到带有GLU（gated linear unit）的模型权重与激活对称，但对于采用GELU的模型并不适用。

3. **知识蒸馏**：

   - 采用基于交叉熵的logit distillation方法，通过teacher model指导student model进行训练。

4. **微调数据集选择**：

   - 选择合适的微调数据集非常重要。如果QAT数据域太窄或者与原始预训练数据分布存在显著不同，则可能会损害模型的性能。

5. **实验效果**：

   - 在LLM-QAT的论文中，作者对LLaMA-7B/13B/30B模型在低至4比特的量化级别上进行了实验。实验结果表明，量化感知训练对模型效果有很大的改进，特别是在低比特的场景。

#### 2）代码实现

以下是一个简化的LLM-QAT代码示例，假设你已经有一个预训练好的大型语言模型，并且使用PyTorch框架。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import LLamaForCausalLM
from quantizers import Quantizer  # 假设你有一个自定义的量化器

# 加载预训练模型和分词器
model_name = "llama-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 定义量化感知训练模型
class QuantAwareLLM(nn.Module):
    def __init__(self, model, quantizer):
        super(QuantAwareLLM, self).__init__()
        self.model = model
        self.quantizer = quantizer

    def forward(self, input_ids, attention_mask=None):
        # 对权重和激活进行量化
        quantized_model = self.quantizer(self.model)
        outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs

# 定义量化器
class SimpleQuantizer:
    def __init__(self, bit_width):
        self.bit_width = bit_width
        self.max_val = 2 ** (bit_width - 1) - 1
        self.min_val = -2 ** (bit_width - 1)

    def quantize(self, tensor):
        return torch.clamp(tensor, self.min_val, self.max_val).round().to(torch.int)

    def dequantize(self, tensor):
        return tensor.to(torch.float) / self.max_val

# 实例化量化器
quantizer = SimpleQuantizer(bit_width=4)

# 实例化量化感知训练模型
quant_aware_model = QuantAwareLLM(model, quantizer)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(quant_aware_model.parameters(), lr=5e-5)

# 数据生成函数
def generate_data(model, tokenizer, max_length=100):
    input_ids = torch.tensor([[tokenizer.encode("<start>")[0]]])
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 生成训练数据
data = [generate_data(model, tokenizer) for _ in range(1000)]  # 假设生成1000条数据

# 将数据转换为模型输入格式
inputs = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
labels = tokenizer(data, padding=True, truncation=True, return_tensors="pt", return_attention_mask=False)["input_ids"]

# 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in zip(inputs["input_ids"], labels):
        input_ids, target_ids = batch
        optimizer.zero_grad()
        
        # 前向传播
        outputs = quant_aware_model(input_ids=input_ids)
        logits = outputs.logits
        
        # 计算损失
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 保存量化后的模型
torch.save(quant_aware_model.state_dict(), "quant_aware_llm.pth")
```

注意：上述代码是一个简化的示例，并没有包含所有细节，例如KV cache的量化、知识蒸馏的具体实现等。在实际应用中，需要更复杂的实现和调优。