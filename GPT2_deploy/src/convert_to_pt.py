import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM
from torch.utils.mobile_optimizer import optimize_for_mobile

# 定义转化后的模型名称
model_ori_pt = 'F:\\llm-deploy\\GPT2_deploy\\gpt2_origin.pt'

# 加载 GPT-2 模型和 tokenizer
model_name_or_path = 'F:\\llm-deploy\\GPT2_deploy\\GPT-2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model_ori = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# 模型在 CPU 上运行
device = torch.device('cpu')
model_ori.to(device)
model_ori.eval()

# 创建一个简化版的 forward 函数，只返回 logits
class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(GPT2Wrapper, self).__init__()
        self.model = model
    
    def forward(self, input_ids):
        outputs = self.model(input_ids)
        # 只返回 logits 部分
        return outputs.logits

# 使用 GPT2Wrapper 包装模型
wrapped_model = GPT2Wrapper(model_ori)

# 创建一个示例输入张量
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# 使用 torch.jit.trace 进行模型转化
traced_model = torch.jit.trace(wrapped_model, input_ids)
traced_model = optimize_for_mobile(traced_model)
# 保存转换后的模型
traced_model._save_for_lite_interpreter(model_ori_pt)

print(f"Model saved to {model_ori_pt}")
