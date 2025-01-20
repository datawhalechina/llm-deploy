# 7-3 推理框架的辅助增值功能
> 不属于框架的核心功能，但让用户用起来比较舒服的一些东西
> 
> 这里还是用 Triton 举例，妥妥的业界标杆

## 7-3-1 Model Analyzer
[Model Analyzer](https://github.com/triton-inference-server/model_analyzer)是一个 CLI 工具，可以帮助模型训练者在给定的硬件上找到最优的模型配置，并且可以生成报告，帮助你了解不同参数设置，或者其他的 trade-off 操作下，模型的性能变化。

首先，对于 Triton-inference-server 支持的模型类型，model_analyzer 也都是支持的，如下所示：
1. Single/Multi-Model
2. Ensemble
3. BLS
4. LLM(大语言模型)

此外，model_analyzer 通过参数搜索完成最优模型配置的寻找，对于参数的搜索，model analyzer 集成了几种 Search Modes，帮助我们简化调参过程
> 参考链接：https://github.com/triton-inference-server/model_analyzer/blob/main/docs/config_search.md

1. Default Search Mode：对于不同模型类型，其 Default Search Mode 是不同的，例如 single 模型而言，其 default search mode 是 Brute Force Search，但是对于 Multi-model 而言，其 Default search mode是 Quick Search
2. Optuna Seach：使用一些超参数优化框架进行启发式扫描，来查找最佳配置
3. Quick Search：快速搜索，运用一些启发式的算法，稀疏的对参数进行搜索
4. Automatic Brute Search：自动暴力搜索
5. Manual Brute Search：手动暴力搜索，手动扫描模型配置中指定的参数

## 7-3-2 Model Navigator
[Model Navigator](https://github.com/triton-inference-server/model_navigator)是一个推理工具包，简化了模型的移动工作，并且提供了很多 [pipeline示例](https://github.com/triton-inference-server/model_navigator/tree/main/examples)

它可以自动执行几个关键步骤，包含模型导出、转换，性能测试和分析，并且可以将生成的优化模型轻松的部署到 Triton Inference Server上，下面简单介绍一下它的 Features

- Model Export and Conversion：自动执行各种格式之间的模型导出和转换过程
- Correctness Testing：正确性测试，确保转换后的模型笨狗产生正确的输出，并进行验证
- Models Depolyment：通过专用的 API 在 PyTrition 和 Triton Inference Server 上自动部署模型或 Pipelines
- Pipelines Optimazation：管道功能，优化了像 Stable Diffusion 和 Whisper 这样的 Pytorch 模型的代码流程
  ```python
    import model_navigator as nav
    from transformers.modeling_outputs import BaseModelOutputWithPooling
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    
    
    def get_pipeline():
        # Initialize Stable Diffusion pipeline and wrap modules for optimization
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        pipe.text_encoder = nav.Module(
            pipe.text_encoder,
            name="clip",
            output_mapping=lambda output: BaseModelOutputWithPooling(**output),
        )
        pipe.unet = nav.Module(
            pipe.unet,
            name="unet",
        )
        pipe.vae.decoder = nav.Module(
            pipe.vae.decoder,
            name="vae",
        )
        return pipe
    ```
    使用 nav.Module包裹模型组件，就可以在数据上完成端到端的优化。例如下面我们准备一个简单的数据加载器：
    ```python
    def get_dataloader():
    # 第一个元素是 batch size
      return [(1, "a photo of an astronaut riding a horse on mars")]
    ```    
    接着，我们执行模型优化并显式的加载最高性能版本
    ```python
  pipe = get_pipeline()
  dataloader = get_dataloader()
  nav.optimize(pipe, dataloader)
  nav.load
    ```
    同样的，也可以使用 pipeline 直接进行优化模型的推理
  ```python
  pipe.to("cuda")
  images = pipe(["a photo of an astronaut riding a horse on mars"])
  image = images[0][0]
  image.save("an_astronaut_riding_a_horse.png")
  ```




    