# 基于In-context learning 蒸馏算法与实现

大模型的一大能力是ICL，即在推理阶段进行少样本学习。然而，由于算力限制，不可能将具有ICL能力的大模型部署到实时推理系统上。有没有一种技术能够将大模型的ICL能力迁移到一个小模型中呢？这就是本节将要探讨的问题。

## ICL 微调
ICL是指输入给模型以下格式的内容：
```
x_1, y_1;
x_2, y_2;
x_3, 
```
具有ICL能力的模型会输出：
```
y_3
```

其中x,y是一对输入，输出示例。这一对输入输出的格式不一定在训练时见过，但是只要在prompt前加几个例子，模型就能学到其中的格式和逻辑，实现了推理时的学习。

ICL微调就是让模型的输出期望的y_3概率尽量大, 这就是损失函数。




## 参考资料
1. [In-context Learning Distillation: Transferring Few-shot Learning Ability of Pre-trained Language Models](http://arxiv.org/abs/2212.10670)

