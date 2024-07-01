# LLM的知识蒸馏

自2017年起，在自然语言处理领域，预训练大模型的参数规模呈指数级别增长，但终端场景的算力比较有限。目前来说，要想调用大模型的功能，需要先把大模型部署到云端或某个数据中心，再通过API远程访问。这种方法会导致网络延迟、依附网络才能应用模型等问题的存在，因此轻量化神经网络具有必要性。模型压缩可以将大型、资源密集型模型转换为适合存储在受限移动设备上的紧凑版本。此外它可以优化模型，以最小的延迟更快地执行，或实现这些目标之间的平衡。

轻量化神经网络分为四个主要的技术路线：

- 压缩已经训练好的大的模型，如知识蒸馏、权值量化、网络剪枝（包括权重剪枝和通道剪枝）、注意力迁移等；
- 直接训练轻量化网络，如squeezeNet, MobileNet, Mnasnet, ShuffleNet, Xception, EfficientNet等；
- 通过一些数值方法加速卷积运算，常用加速方法包括winograd卷积、低秩分解等；
- 硬件部署。衡量一个轻量化网络是否足够好有以下一些角度：参数量、计算量、内存访问量、耗时、能耗等。有而知识蒸馏属于上述的第一种方法。

<div align=center>
<img src="https://github.com/gyfffffff/llm-deploy/blob/main/%E6%A8%A1%E5%9E%8B%E8%92%B8%E9%A6%8F/1.%20%E8%92%B8%E9%A6%8F%E5%9F%BA%E7%A1%80/chapter3_1/images/Figure%204.png" width="700">
</div>

当使用LLM作为教师网络时，根据是否强调将LLM的涌现能力（Emergent Abilities, EA）蒸馏到小型语言模型（SLM）中来进行分类，可以把知识蒸馏分为两类：标准知识蒸馏（Standard KD）和基于涌现能力的知识蒸馏（EA-based KD）。

    涌现能力是指在较小的模型中不出现，而在较大的模型中出现的能力，则可以称之为“涌现能力“；
    比如与 BERT（330M）和 GPT-2（1.5B）等较小模型相比，GPT-3（175B）和 PaLM（540B）等 LLM 展示了独特的行为，这些LLM在处理复杂的任务时表现出令人惊讶的能力。
    (An ability is emergent if it is not present in smaller models but is present in larger models.)。

标准知识蒸馏（Standard KD）旨在使学生模型学习LLM所拥有的常见知识，如输出分布和特征信息。这种方法类似于传统的知识蒸馏，但区别在于教师模型是LLM。相比之下，基于涌现能力的知识蒸馏（EA-based KD）不仅仅是将LLM的常见知识转移到学生模型中，还涵盖了蒸馏它们独特的涌现能力。具体来说，基于涌现能力的知识蒸馏（EA-based KD）又分为了上下文学习（ICL）、思维链（CoT）和指令跟随（IF）。

<div align=center>
<img src="https://github.com/gyfffffff/llm-deploy/blob/main/%E6%A8%A1%E5%9E%8B%E8%92%B8%E9%A6%8F/1.%20%E8%92%B8%E9%A6%8F%E5%9F%BA%E7%A1%80/chapter3_1/images/Figure%205.png" width="500">
</div>

有关分类模型的蒸馏、小模型蒸馏的开源项目可以参考：
https://github.com/datawhalechina/awesome-compression/blob/main/docs/ch06/ch06.md

根据蒸馏方法可以把知识蒸馏分为两类：一种是黑盒知识蒸馏(Black-box KD)，一种是白盒知识蒸馏(White-box KD)。黑盒知识蒸馏将教师网络的预测结果当成训练集输入学生网络，而白盒知识蒸馏不仅利用了教师网络的预测结果，还把教师网络的参数输入轻量级模型。

<div align=center>
<img src="https://github.com/gyfffffff/llm-deploy/blob/main/%E6%A8%A1%E5%9E%8B%E8%92%B8%E9%A6%8F/1.%20%E8%92%B8%E9%A6%8F%E5%9F%BA%E7%A1%80/chapter3_1/images/Figure%206.png" width="500">
</div>

参考文献： 

[1] Knowledge Distillation [https://arxiv.org/pdf/1503.02531.pdf]

[2] Do Deep Nets Really Need to be Deep? [https://arxiv.org/pdf.1312.6184.pdf]

[3] A Servey on Model Compression for Large Language Models [https://arxiv.org/pdf/2308.07633.pdf] 
