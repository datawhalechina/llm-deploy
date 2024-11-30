# 2.1 蒸馏

本章将介绍大模型的主流蒸馏方法和代码。

## Roadmap
### 1. 蒸馏基础

- 1.1 为什么要做LLM蒸馏
- 1.2 和小模型蒸馏的不同
- 1.3 LLM蒸馏的分类
     - a. 白盒
     - b. 黑盒


基于分类模型的蒸馏、小模型蒸馏，这部分请参考datawhale另一个关于模型压缩的项目[https://github.com/datawhalechina/awesome-compression/blob/main/docs/ch06/ch06.md](https://github.com/datawhalechina/awesome-compression/blob/main/docs/ch06/ch06.md)）

### 2. 标准知识蒸馏（白盒蒸馏）
-  2.1 概述
      - 何时使用白盒蒸馏
-  2.2 MiniLLM
-  2.3 BabyLlama

### 3. 基于涌现能力的蒸馏（黑盒蒸馏）
-  3.1 概述
      - 什么是涌现能力
      - 与标准蒸馏的不同
      - 何时使用黑盒蒸馏
- 3.2 基于In-context learning 蒸馏算法与实现

- 3.3 基于CoT蒸馏算法与实现-找一种作为代码例子简单实现的

- 3.4 指令跟随蒸馏算法与实现


### 4. 总结
