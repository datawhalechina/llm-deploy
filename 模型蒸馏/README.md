# 模型蒸馏

本章讲介绍基于Transformer的模型的主流蒸馏方法和代码，还将实现一个端侧部署demo。

## Roadmap
### 1. 蒸馏基础

- 为什么要做LLM蒸馏
- 和小模型蒸馏的不同
- LLM蒸馏的分类
     - 白盒
     - 黑盒


基于分类模型的蒸馏、小模型蒸馏，这部分请参考datawhale另一个关于模型压缩的项目[https://github.com/datawhalechina/awesome-compression/blob/main/docs/ch06/ch06.md](https://github.com/datawhalechina/awesome-compression/blob/main/docs/ch06/ch06.md)）

### 2. 标准知识蒸馏（白盒蒸馏）
-  概述
      - 何时使用白盒蒸馏
-  MiniLLM
-  GKD

### 3. 基于涌现能力的蒸馏（黑盒蒸馏）
-  概述
      - 什么是涌现能力
      - 与标准蒸馏的不同
      - 何时使用黑盒蒸馏
- 基于In-context learning 蒸馏算法与实现

- 基于CoT蒸馏算法与实现-找一种作为代码例子简单实现的

- 指令跟随蒸馏算法与实现


### 4. 总结
- 前沿相关工作扩展
- 总结

## 参与贡献

- 如果你想参与到项目中来欢迎查看项目的 [Issue]() 查看没有被分配的任务。
- 如果你发现了一些问题，欢迎在 [Issue]() 中进行反馈🐛。
- 如果你对本项目感兴趣想要参与进来可以通过 [Discussion]() 进行交流💬。

如果你对 Datawhale 很感兴趣并想要发起一个新的项目，欢迎查看 [Datawhale 贡献指南](https://github.com/datawhalechina/DOPMC#%E4%B8%BA-datawhale-%E5%81%9A%E5%87%BA%E8%B4%A1%E7%8C%AE)。

## 贡献者名单

| 姓名 | 职责 |  |
| :----| :---- | :---- |
| Yufei | 项目负责人/第2节贡献者 |  |
| Zhixiong | 第3节贡献者 |  |
| Xiaoyu | 第1节贡献者 |  |



## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
