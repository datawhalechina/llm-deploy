## 1.2：经典的量化方法（How-part1:PTQ）
第二章讲经典的PTQ量化LLM.int8，SQ，GPTQ（涵盖weight-only 和 weight-act）-方式：原理讲解+代码，

### 1.2.1 PTQ量化LLM.int8方式

PTQ（Post-Training Quantization）是一种在模型训练后进行的量化方法，通过这种方法可以在不重新训练模型的情况下，将模型的权重和激活值从浮点表示（如FP32）转换为低精度表示（如INT8），从而减少模型的存储大小和计算需求，提高推理性能。

#### 1）INT8原理讲解

PTQ量化的核心思想是将浮点数通过缩放因子（scale）映射到整数范围内，从而减少存储和计算开销。在INT8量化中，通常将浮点数映射到[-128, 127]范围内的8位整数。

量化过程包括以下几个步骤：

- **确定缩放因子**：缩放因子（scale）是通过浮点数中的最大值和最小值计算得到的，它决定了浮点数到整数的映射关系。
- **量化**：将浮点数除以缩放因子，四舍五入到最近的整数，并限制在[-128, 127]范围内。
- **反量化**：在推理过程中，需要将量化后的整数重新转换为浮点数，以便进行后续计算。反量化是通过将量化后的整数乘以缩放因子来实现的。

在实际应用中，除了权重外，还需要对激活值进行量化。激活值的量化通常在推理过程中动态进行，即在每一层计算前将激活值量化为INT8，然后在计算完成后将结果反量化为FP32，以便传递给下一层。

#### 2）INT8代码实现

以下是一个简单的PTQ量化INT8的示例代码，使用PyTorch框架：

```python
import torch
import torch.nn as nn
import torch.quantization as quant
import torch.optim as optim
import torch.nn.functional as F

# 定义一个简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 初始化模型
model = SimpleCNN()

# 准备校准数据集（用于收集统计信息）
calibration_dataloader = ...  # 这里需要替换为实际的校准数据集加载器

# 模型校准
model.eval()
with torch.no_grad():
    for inputs, _ in calibration_dataloader:
        model(inputs)

# 准备量化配置
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)

# 转换模型为量化模式
model.cpu()
quant.convert(model, inplace=True)

# 保存量化后的模型
torch.save(model.state_dict(), 'quantized_model.pth')
```

在实际应用中，量化过程通常更加复杂，需要仔细处理模型的每一层，以确保量化后的模型能够正确运行并保持良好的性能。

### 1.2.2 SmoothQuant（SQ）量化方式

SmoothQuant是一种旨在减少量化过程中信息损失的量化方法。它通过平滑量化过程，使得量化后的模型能够更好地保留原始模型的性能。

#### 1）SQ原理讲解

SmoothQuant的核心思想是在量化过程中引入平滑性约束，以减少量化误差。这通常通过优化一个包含量化误差和平滑性约束的损失函数来实现。

在SmoothQuant中，通常使用以下步骤进行量化：

- **收集统计信息**：使用校准数据集收集模型的权重和激活值的统计信息，如最大值、最小值等。
- **计算量化参数**：根据收集到的统计信息，计算量化所需的缩放因子和零点（zero-point）。
- **平滑量化**：在量化过程中引入平滑性约束，以减少量化误差。这可以通过在量化损失函数中添加平滑性正则项来实现。
- **模型校准**：使用量化后的模型和校准数据集进行微调，以优化量化效果。

#### 2）SQ代码实现

SmoothQuant的具体实现通常依赖于特定的量化框架和库。以下是一个简化的示例代码，展示了如何在PyTorch中实现类似SmoothQuant的量化过程：

```python
import torch
import torch.nn as nn
import torch.quantization as quant
import torch.optim as optim

# 定义一个简单的神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 初始化模型
model = SimpleNet()

# 准备校准数据集（用于收集统计信息）
calibration_dataloader = ...  # 这里需要替换为实际的校准数据集加载器

# 收集权重和激活值的统计信息
def collect_stats(model, dataloader):
    act_max = {}
    act_min = {}
    for inputs, _ in dataloader:
        model(inputs)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.detach().cpu().numpy()
                act_max[name + '.weight'] = np.max(weight)
                act_min[name + '.weight'] = np.min(weight)
                # 注意：这里省略了激活值的统计，实际中需要添加
    return act_max, act_min

act_max, act_min = collect_stats(model, calibration_dataloader)

# 准备量化配置
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)

# 自定义量化函数，引入平滑性约束
def smooth_quantize(tensor, scale, zero_point, smooth_factor=0.1):
    quantized = torch.round(tensor / scale) - zero_point
    quantized = torch.clamp(quantized, min=-128, max=127).to(tensor.dtype)
    smoothed = tensor + smooth_factor * (quantized.to(tensor.dtype) * scale + zero_point - tensor)
    return quantized, smoothed

# 对模型进行量化，并引入平滑性约束
def quantize_model(model, act_max, act_min):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            scale = (act_max[name + '.weight'] - act_min[name + '.weight']) / 255.0
            zero_point = 0  # 简化处理，实际中需要计算
            quantized_weight, smoothed_weight = smooth_quantize(
                torch.tensor(weight, dtype=torch.float32), scale, zero_point
            )
            module.weight.data = torch.tensor(smoothed_weight, dtype=torch.float32).to(module.weight.device)
            # 注意：这里省略了偏置和激活值的量化，实际中需要添加
    return model

quantized_model = quantize_model(model, act_max, act_min)

# 保存量化后的模型
torch.save(quantized_model.state_dict(), 'smooth_quantized_model.pth')
```

请注意，上述代码是一个简化的示例，用于说明如何在量化过程中引入平滑性约束。在实际应用中，量化过程通常更加复杂，需要仔细处理模型的每一层，并确保量化后的模型能够正确运行并保持良好的性能。

### 1.2.3 GPTQ量化方式

GPTQ（Gradient-based Post-Training Quantization）是一种基于梯度的训练后量化方法。它通过最小化量化前后模型输出的误差来优化量化参数，从而提高量化模型的性能。

#### 1）原理讲解
GPTQ的核心在于最小化量化引入的输出误差，它通过逐层处理模型的权重矩阵，并利用一小部分校准数据来最小化量化前后模型输出的差异。

1. **收集校准数据**：
   GPTQ从训练数据或相关数据集中抽取一小部分样本作为校准数据。这些数据用于在后量化过程中评估量化误差，并帮助优化量化参数。

2. **逐层处理**：
   GPTQ对模型的每一层进行独立量化，避免全局优化的复杂度。逐层量化允许对不同层采用不同的量化策略，以最小化量化带来的误差。

3. **最小化输出误差**：
   对于每一层，GPTQ寻找最佳的量化权重，使得在校准数据上的输出误差最小。这通常涉及对量化误差进行建模，并通过优化算法找到最优的量化参数。

4. **更新权重**：
   将量化后的权重替换原始权重，完成量化过程。

GPTQ的量化过程可以分为以下几个步骤：

- **计算Hessian矩阵**：
  Hessian矩阵是二阶导数矩阵，用于描述损失函数相对于模型参数的二阶变化率。GPTQ利用Hessian矩阵来估计量化误差，并优化量化参数。

- **逐层weight量化**：
  GPTQ对每一层的权重矩阵进行量化。量化过程涉及将浮点数权重转换为低比特的整数表示，同时尽可能保持模型的性能。

- **保存量化weight**：
  完成量化后，GPTQ保存量化后的权重，以便在推理时使用。

在量化过程中，GPTQ采用了多种技术来减少量化误差并提高模型性能：

- **误差最小化原理**：
  GPTQ通过最小化量化引入的输出误差来实现高精度量化。这通常涉及对量化误差进行建模，并通过优化算法找到最优的量化参数。

- **逐列优化**：
  为了降低计算复杂度，GPTQ采用了逐列优化的方法。将权重矩阵的列表示为wi，对每一列进行量化，同时考虑之前列量化引入的误差累积。

- **量化策略**：
  GPTQ可以采用多种量化策略，如对称量化、非对称量化、均匀量化等。同时，量化器需要满足硬件的限制，确保量化后的值在表示范围内。

- **量化感知训练（Quantization-Aware Training, QAT）**：
  在训练模型时，GPTQ模拟了量化过程，使模型能够在训练时就适应量化的精度限制，从而减少推理时的性能损失。这种方法通常在训练结束的最后阶段进行。

#### 2）GPTQ的代码实现

以下是一个基于PyTorch和Transformers库的GPTQ量化示例代码。这个示例代码演示了如何对GPT-2模型进行4-bit量化，并保存量化后的模型。

```python
# 安装必要的库
!pip install transformers
!pip install accelerate
!pip install auto-gptq  # 假设auto-gptq是一个提供GPTQ功能的库，实际中可能需要使用其他库或自行实现GPTQ功能

# 导入必要的模块
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # 假设auto_gptq提供了GPTQ的量化配置和模型

# 指定模型名称或路径
model_name_or_path = "gpt2"

# 定义量化配置
quantize_config = BaseQuantizeConfig(
    bits=4,                # 量化到4-bit
    group_size=128,        # 分组大小，通常为128或None
    desc_act=False,        # 是否禁用激活函数的量化
)

# 加载模型并进行量化
model = AutoGPTQForCausalLM.from_pretrained(
    model_name_or_path,
    quantize_config=quantize_config,
    use_triton=False  # 如果安装了Triton加速器，可设为True
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# 保存量化后的模型
save_directory = "gpt2-quantized"
model.save_quantized(save_directory)
tokenizer.save_pretrained(save_directory)

# 推理测试
# 加载量化后的模型
model_quantized = AutoGPTQForCausalLM.from_quantized(
    save_directory,
    use_safetensors=True,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    use_triton=False,
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=True)

# 准备输入
input_text = "今天天气如何?"
inputs = tokenizer(input_text, return_tensors="pt")

# 将输入移动到模型设备
inputs = inputs.to(model_quantized.device)

# 生成输出
with torch.no_grad():
    output_ids = model_quantized.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
    )

# 解码输出
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
```

**注意**：上述代码中的`auto_gptq`库和`AutoGPTQForCausalLM`类是假设存在的，实际中可能需要使用其他库或自行实现GPTQ功能。此外，代码中的量化配置和模型加载方式也可能需要根据实际使用的库和模型进行调整。

在实际应用中，GPTQ的量化过程可能涉及更多的细节和优化。例如，可能需要计算Hessian矩阵的逆矩阵，这通常是一个计算量很大的操作。为了降低计算复杂度，GPTQ可能会采用一些近似方法或优化算法来加速量化过程。
