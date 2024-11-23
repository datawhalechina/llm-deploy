### 5.4 TensorFlow框架专用模型表示

`TensorFlow` 是另一种广泛使用的深度学习框架，其提供了专用的模型保存和部署格式 `SavedModel`。`TensorFlow SavedModel` 是一种灵活且全面的格式，支持模型的训练、评估和推理。



#### 5.4.1 TensorFlow SavedModel简介

`TensorFlow SavedModel` 是 `TensorFlow` 中的标准模型格式，能够保存整个模型，包括图结构、权重和元数据。与其他模型格式相比，`SavedModel` 能够更好地支持分布式训练和跨平台部署。

 `SavedModel` 包含以下几个主要部分：

- `assets`：包含模型所需的外部文件，如词汇表。
- `variables`：包含模型的可训练变量。
- `saved_model.pb`：包含模型的图定义和元数据。

```python
import tensorflow as tf

# 定义一个简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(data, labels, epochs=10)

# 保存模型
model.save("./saved_model", save_format="tf")  # 保存SavedModel格式
model.save("SPEED_model.h5", save_format="h5")  # 保存h5格式

# 加载模型
mymodel = tf.saved_model.load("./saved_model")

```





#### 5.4.2 模型转换

将模型转换为 `TensorFlow SavedModel` 格式非常简单，可以直接调用 `save` 方法。与此同时，`TensorFlow` 还提供了从其他格式（如 `HDF5`）转换为 `SavedModel` 的工具。

```python
# 将 Keras 模型保存为 HDF5 格式
model.save("model.h5")

# 从 HDF5 格式加载模型并转换为 SavedModel 格式
loaded_model = tf.keras.models.load_model("model.h5")
loaded_model.save("converted_saved_model")

```





#### 5.4.3 适用场景和限制分析

`TensorFlow SavedModel` 格式适用于多种场景：

- **跨平台部署**：`SavedModel` 支持在不同的平台和设备上运行，如 `TensorFlow Serving`、`TensorFlow Lite` 和 `TensorFlow.js`。
- **分布式训练**：`SavedModel` 可以保存和加载用于分布式训练的模型。
- **图优化**：`SavedModel` 支持通过 `TensorFlow` 的优化工具对模型进行优化，以提高推理性能。

然而，`SavedModel` 格式也有一些限制：

- **文件大小**：由于保存了完整的图结构和权重，`SavedModel` 文件可能会比较大。
- **依赖性**：某些外部依赖可能无法包含在 `SavedModel` 中，需要额外管理。



#### 5.4.4 pytorch，tensorflow和onnx框架模型表示的相互转换

在深度学习模型开发过程中，可能需要在不同框架之间转换模型表示格式。`ONNX` 提供了一种通用的中间表示，支持在 `PyTorch` 和 `TensorFlow` 之间进行模型转换。

##### `PyTorch` 转换为 `ONNX`：

```python
import torch
import torch.onnx

# 定义并训练一个简单的 PyTorch 模型
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = torch.nn.Linear(32, 10)
        self.fc2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
dummy_input = torch.randn(1, 32)

# 导出为 ONNX 格式
torch.onnx.export(model, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])

```

##### `ONNX` 转换为 `TensorFlow`：

```python
import onnx
from onnx_tf.backend import prepare

# 加载 ONNX 模型
onnx_model = onnx.load("model.onnx")

# 将 ONNX 模型转换为 TensorFlow 格式
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model.pb")

```

##### `TensorFlow` 转换为 `ONNX`：

```python
import tf2onnx
import tensorflow as tf

# 定义并训练一个简单的 TensorFlow 模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(data, labels, epochs=10)

# 将 TensorFlow 模型转换为 ONNX 格式
spec = (tf.TensorSpec((None, 32), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

```

在本章中，我们详细介绍了 `TensorFlow` 框架专用的模型表示 `SavedModel` 及其适用场景和限制分析，同时讨论了 `PyTorch`、`TensorFlow` 和 `ONNX` 之间的模型转换方法。
