# 并行

## 概述

## 数据并行

## 流水线并行


当前主流大模型的参数规模通常达到数十亿甚至成百上千亿级，单块 GPU 很难加载整个模型，因此我们需要将模型分布在不同设备上，这也称为模型并行方法。模型并行方法有两种：流水线并行和张量并行。如下图所示，流水线并行将模型按层划分到不同 GPU 上，在各层之间进行并行计算；

![Untitled](figs/Untitled.png)


## 张量并行

与流水线并行不同，张量并行是将模型中的张量进行拆分然后分配到不同的 GPU 上，每块 GPU 都可以得到所有层张量的部分参数。这样在前向计算中有效减少了流水行并行中的空置时间，提高了 GPU 的显存利用率，因此张量并行也成了当下大模型训练和推理的主流并行方法。显存效率：模型并行会根据 worker 数量成比例地减少显存使用量。至关重要的是，这是减少单个网络层的激活显存的唯一方法。DeepSpeed 通过在模型并行 worker 之间划分激活显存来进一步提高显存效率。
计算效率：由于每次前向和反向传播中都需要额外通信激活值，模型并行的计算效率很低。模型并行需要高通信带宽，并且不能很好地扩展到通信带宽受限的节点。此外，每个模型并行worker 都会减少每个通信阶段之间执行的计算量，从而影响计算效率。模型并行性通常与数据并行性结合使用，以在内存和计算效率之间进行权衡。

### 1D 张量并行

从实现上看，张量并行就是把同一层的参数矩阵分块进行相互独立的矩阵乘法计算，然后合并结果，通过不同 GPU 之间的通信，保证计算图的正确性。对一个单独的矩阵，我们可以很自然想到基于行和列进行拆分，称为行并行和列并行。

如图 (a) 所示，以一般矩阵乘法为例，假设我们有 $Y=XW$ ，其中 $X\in\mathbb R^{2\times2}$ 为输入数据，$W\in\mathbb R^{2\times2}$ 为参数矩阵，在两块 GPU 上进行并行计算，输入数据 $X$ 与权重向量 $W$ 进行矩阵相乘时，计算行列对之间的点积是相互独立的。列并行就是将权重参数 $W$ 沿列分割成 $W=[W_0\ W_1]$，每块 GPU 持有一列参数 $W_0, W_1 \in \mathbb R^{2\times1}$，如图 (b) 所示，我们将 $X$ 分别输入 rank 0 和 rank 1 的 GPU 上，然后与 GPU 上的参数进行矩阵相乘，我们将得到 $Y_0, Y_1$ ，然后通过一次拼接操作就可以得到和 (a) 等价的 $Y$。而行并行则是将 $W$ 沿行进行切分 $W=[W_0 \ W_1]^{\mathrm T}$ 放置，每块 GPU 持有一行参数 $W_0, W_1 \in \mathbb R^{1\times2}$ ，然后将输入 $X$ 也沿列切分为 $X_0, X_1 \in \mathbb R^{2\times1}$ 并输入到两块 GPU 上分别进行矩阵乘法运算得到 $Y^\prime,Y^{\prime\prime}\in\mathbb R^{2\times 2}$ ，然后按元素位置相加 $Y^\prime+Y^{\prime\prime}$ 也可以得到等价的 $Y$。

![Untitled](figs/Untitled%201.png)

![Untitled](figs/Untitled%202.png)

![Untitled](figs/Untitled%203.png)

不难看出，无论行并行还是列并行，想要得到最终结果，我们都需要收集所有 GPU 上的计算结果，而神经网络通常包含很多层，也就意味着在进行当前层的张量并行时，我们必须确保所有 GPU 都接收到上一层前向计算的完整计算图结果，而将所有设备的结果统筹收集的过程，称为全收集操作 All Reduce，这也正是张量并行对设备间通信带宽要求较高的原因。而我们可以通过灵活使用行并行和列并行，尽量减少前向计算中的全收集操作。

观察行并行和列并行的输入输出形式，不难发现在不进行全收集操作的情况下，行并行的输出形式恰好是列并行的输入形式，列并行的输出方式恰好是行并行的输入形式，因此不难想到，在前向计算中，这两种并行方式是交替使用的。以 Transformer 中的 FFN 层为例，FFN 层包含两个线性层 nn.linear 以及线性层中间的激活函数 GELU(·)，假设两个线性层参数分别为 $A$ 和 $B$。如果在第一个线性层进行行并行将 $A$ 拆分为 $A=[A_0 \ A_1]^{\mathrm T}$，由于激活单元需要完整计算结果，所以就需要进行一次全收集操作从而得到经过第一个线性层后的完整计算图结果 $Y=Y^\prime+Y^{\prime\prime}$，再经过激活函数 GELU，而在第二个线性层处无论进行行并行还是列并行，都需要在输入下个 Transformer 层的多注意力模块时进行全收集操作，这样在经过 FFN 层时就需要两次全收集操作；而如果在第一个线性层进行列并行 $A = [A_0 \ A_1]$，那么得到的计算图结果 $Y_0=[y_{00} \ y_{01}]^{\mathrm T}, Y_1=[y_{10} \ y_{11}]^{\mathrm T}$ 是可以分别独立经过激活单元的，而后在第二个线性层处进行行并行 $B=[B_0 \ B_1]^{\mathrm T}$，这样就只需要一次全收集操作 $Z=Z^\prime+Z^{\prime\prime}$，下图分别展示了 FFN 层两种张量并行计算步骤。

![Untitled](figs/Untitled%204.png)

### 2D 和 2.5D 张量并行

Nvidia Megatron-LM 使用的是 1D 张量并行，这种方法虽然将参数划分到多个处理器上，但每个处理器仍需要存储整个中间计算图结果，在每次计算中，每块 GPU 都需要与其他设备通信，在处理大模型时会浪费大量显存空间，通信成本会不断增加。对此，Colossal-AI 提供多维张量并行，这里先介绍 2D 张量并行。2D 张量并行技术将输入数据、模型权重和层输出拆分成两个维度，与 1D 张量并行相比，内存消耗更低。对于一个 2×2 的矩阵乘法，假设我们有 4 块 GPU，那就可以将矩阵乘法分块到每块 GPU 上。将输入 $X$ 和参数 $W$ 进行如下分块 $X=[X_0\ X_1]$, $W=[W_0\ W_1]^{\mathrm T}$，首先在 Step 1 进行 $X_0$ 与 $W_0$ 的矩阵乘法，将4 个算子分配到 4 块 GPU 上进行计算，同样，Step 2 进行 $X_1$ 与 $W_1$ 的运算，最后将两步的计算结果相加，便得到最终结果。

![Untitled](figs/Untitled%205.png)

```python
import colossalai  
import colossalai.nn as col_nn  
import torch  
from colossalai.utils import print_rank_0
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import get_current_device

# 并行设置
CONFIG = dict(parallel=dict(
    data=1,
    pipeline=1,
    tensor=dict(size=4, mode='2d'),
))

parser = colossalai.get_default_parser()  
    colossalai.launch(config=CONFIG,  
    rank=args.rank,  
    world_size=args.world_size,  
    local_rank=args.local_rank,  
    host=args.host,  
    port=args.port)  
  
class MLP(torch.nn.Module):  
    def __init__(self, dim: int = 256):  
        super().__init__()  
        intermediate_dim = dim * 4  
        self.dense_1 = col_nn.Linear(dim, intermediate_dim)  
        print_rank_0(f'Weight of the first linear layer: {self.dense_1.weight.shape}')  
        self.activation = torch.nn.GELU()  
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)  
        print_rank_0(f'Weight of the second linear layer: {self.dense_2.weight.shape}')  
        self.dropout = col_nn.Dropout(0.1)  

    def forward(self, x):  
        x = self.dense_1(x)  
        print_rank_0(f'Output of the first linear layer: {x.shape}')  
        x = self.activation(x)  
        x = self.dense_2(x)  
        print_rank_0(f'Output of the second linear layer: {x.shape}')  
        x = self.dropout(x)  
        return x

# 创建模型
m = MLP()

# 随机输入一些数据来运行这个模型
x = torch.randn((16, 256), device=get_current_device())

# partition input
torch.distributed.broadcast(x, src=0)
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_COL)]
x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2D_ROW)]
print_rank_0(f'Input: {x.shape}')

x = m(x)
```

2.5D张量并行技术通过为矩阵添加可选的深度维度，使2D并行技术向前推进了一步。当扩展到大量设备时，这种方法可进一步减小通信量。张量被分割，使得N=S2D，其中S是正方形一侧的大小，D是立方体的深度。当深度为1时，这种方法与2D张量并行技术类似。之所以叫 2.5D 张量并行是因为在 d = 1 时，这种并行模式可以退化成 2D 张量并行；在 d = q 时，它就变成了3D 张量并行。下面我们来看看 3D 张量并行。

```python
# 并行设置
CONFIG = dict(parallel=dict(  
    data=1,  
    pipeline=1,  
    tensor=dict(size=8, mode='2.5d', depth=2),  
))

...
  
# 创建模型
m = MLP()

# 随机输入一些数据来运行这个模型
x = torch.randn((16, 256), device=get_current_device())

# partition input  
torch.distributed.broadcast(x, src=0)  
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_DEP)]  
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_COL)]  
x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_2P5D_ROW)]  
print_rank_0(f'Input: {x.shape}')  
  
x = m(x)
```

### 3D 张量并行

3D张量并行技术将张量分割成立方体形状，并对第一个和最后一个维度进行划分。这种技术可实现最佳通信成本，并均匀分配计算和内存使用量。当扩展到更多设备时，高级张量并行技术可进一步减小通信量，并与流水线并行更加兼容。与1D张量并行技术不同，1D张量并行技术是在将张量传递到下一个流水线阶段之前将其分割成若干块，并在到达目标流水线阶段后通过节点内通信进行收集，而高级张量并行技术已经为下一阶段提供了整个逻辑张量的子块，因此无须进行拆分-收集操作，从而降低通信成本。

```python
# 并行设置
CONFIG = dict(parallel=dict(  
    data=1,  
    pipeline=1,  
    tensor=dict(size=8, mode='3d'),  
))

...
  
# 创建模型
m = MLP()

# 随机输入一些数据来运行这个模型
x = torch.randn((16, 256), device=get_current_device())

# partition input  
torch.distributed.broadcast(x, src=0)  
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_WEIGHT)]  
x = torch.chunk(x, 2, dim=0)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_INPUT)]  
x = torch.chunk(x, 2, dim=-1)[gpc.get_local_rank(ParallelMode.PARALLEL_3D_OUTPUT)]  
print_rank_0(f'Input: {x.shape}')  
  
x = m(x)
```