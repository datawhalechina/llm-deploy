# MiniLLM

大模型能力的强大也伴随着参数量的膨胀，为了以合理的成本部署大模型，如何将大模型的知识蒸馏到小模型是一个问题。传统的知识蒸馏是面向分类等有限状态空间设计的，通过最小化前向KL散度，就能够让学生模型（小模型）学到有限的状态空闲（比如有限的类别）。但是大语言模型本质上做的是自回归式生成任务，传统的知识蒸馏方法不再适用。

MiniLLM是一种针对生成式语言模型的全新的KD方法，是一种白盒蒸馏方法，让学生模型生成更精确可靠的内容，从而实现高性价比的LLM部署落地。



## KL散度
了解传统知识蒸馏的同学应该对KL散度并不陌生，这里我们再复习一下它的概念：
老师分布为p, 学生分布为$q_\theta$, 

前向KL散度可以看成是定义了分布之间的一种除法，p除以q，是 $KL(p||q_\theta) = \sum_i p(i)log\frac{p(i)}{q_\theta(i)}$。一般都要最小化KL散度。

从定义可以看出，在p分布为0的地方，q分布无论为多少，都不影响这一项为0，所以前向KL散度的一个特点是p会在老师分布小的地方比较大。对应到大模型生成上，就是在老师模型输出可能性很小的地方，学生模型却放大了这种可能性，显然这是不对的。

## 逆向KL散度

reversed KL:

$$KL(q_\theta ||p) = \sum_i q_\theta(i)log\frac{q_\theta(i)}{p(i)} = -\sum_i q_\theta(i)log\frac{p(i)}{q_\theta(i)}$$


## 基于策略梯度的优化

## 实战


## 结论
- MiniLLM的优势：
- MiniLLM的局限性：
- 进展更新：

## 参考资料
- MiniLLM: Knowledge Distillation of Large Language Models
- https://github.com/microsoft/LMOps/tree/main/minillm 