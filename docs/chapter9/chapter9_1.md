# 9.1 异步

在预填充阶段，模型需要接收和处理来自不同用户的多个输入。这时可以利用**异步处理**技术来提高效率。

## 流式处理(Streaming)

流式处理采用异步通信方式，减少了不必要的等待和阻塞，提高了系统的并发性能和吞吐量

想象一下,如果用户每次与 ChatGPT 对话时都必须等待整个回答生成完毕再显示结果，这种体验多少不太舒服。因为人类在交流时往往是实时获取信息的，逐字逐句地接收信息更符合自然的交流习惯。

流式处理巧妙地解决了这个问题。它让 LLM 能够"边思考边说话，就像人类交谈一样自然。当模型生成第一个词时，它就立即被发送给用户，紧接着是第二个词，第三个词...这样，用户几乎可以实时地看到回答的形成过程，大大提升了交互体验。

流式处理的魅力不仅仅在于提升用户体验。从技术角度来看，它还大大提高了系统的并发性能。传统的方法可能需要等待整个响应生成完毕才能处理下一个请求，而流式处理允许系统同时处理多个请求的不同部分。想象一下，这就像是一个高效的多任务处理器，能够同时应对多个对话，每个对话都在稳步推进。

vLLM 利用异步生成器 (async generator) 来实现流式输出。这允许模型一边生成 token,一边将其 yield 给调用者,而不需要等待整个响应生成完毕,下面是一个简化的实现:

```python
import asyncio
import random

async def async_word_generator(sentence):
    words = sentence.split()
    for word in words:
        # 模拟推理时间
        await asyncio.sleep(random.uniform(0.3, 1.0))
        yield word

async def stream_sentence():
    sentence = "vLLM 的异步生成器可以实现流式输出 让用户体验更加流畅 同时提高系统效率"
    async for word in async_word_generator(sentence):
        print(word, end=' ', flush=True)
    print()  

async def main():
    await stream_sentence()

if __name__ == "__main__":
    asyncio.run(main())
```

此外,vLLM 官方给出的 API 服务器示例使用了 FastAPI 框架,它原生支持异步编程。这使得服务器可以高效地处理多个并发的流式请求。

不过虽然对用户来说是流式的,但在底层 vLLM 仍然使用批处理来提高效率。它会预先生成一批 token,然后逐个返回给用户。

### Server-Sent-Event

Server-Sent Events (SSE) 是一种允许服务器向客户端推送数据的 Web 技术,ChatGPT 网页应用就使用了它。它的工作原理相对简单但非常有效，让我们来深入了解一下:

- 连接建立: 客户端通过常规的 HTTP 请求与服务器建立连接。这个请求通常包含一个特殊的头部 "Accept: text/event-stream"，告诉服务器客户端希望接收事件流。

- 服务器响应: 服务器以 "Content-Type: text/event-stream" 响应，表明这是一个事件流。然后，服务器保持这个连接开放。

- 数据传输: 服务器可以通过这个开放的连接持续发送数据。每条消息以 "data:" 开头，以两个换行符 "\n\n" 结束

在 FastAPI 中实现 SSE
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def generate_messages():
    messages = ["大语言", "模型", "真的是", "非常", "有趣"]
    for message in messages:
        yield f" {message}\n\n"
        # 模拟推理时间
        await asyncio.sleep(0.5)
    yield " END\n\n"

@app.get("/sse")
async def sse_endpoint():
    return StreamingResponse(generate_messages(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

打开 `http://localhost:8000/sse` 就可以看到 SSE 的效果。

这张图展示了在一个多设备（如多GPU）环境下，如何通过异步通信和计算来优化模型的计算过程。具体来说，它描述了在每个设备上处理一层神经网络时的时间线，并展示了如何在计算过程中隐藏通信开销。

图中主要元素解释：
Scatter 和 Sparse Op：

图中展示了每一层（Layer 1, Layer 2, …, Layer L）都有一个 Scatter 操作和一个 Sparse Op 操作。
Scatter 操作通常是将数据分散到不同的设备上，以便进行并行处理。
Sparse Op 是稀疏操作，可能是指在稀疏矩阵上的计算，这种计算通常只涉及到非零元素，因此可以通过分散计算提高效率。
AllGather：

AllGather 是一个常见的通信操作，用于在所有设备之间收集数据。每个设备将它们的部分计算结果传递给其他设备，从而使所有设备都能够得到完整的数据集。
在图中，AllGather 操作是在每一层的计算后进行的，用来收集每个设备的计算结果。
Comm. (Communication)：

Comm. 表示设备之间的通信。图中展示了通信操作是异步进行的，即通信和计算是重叠的，通信开销被隐藏在计算过程之中。
这种异步通信可以显著减少通信开销对整体计算速度的影响，因为通信操作在设备进行计算时已经开始，并且可以在计算结束之前完成。
图的整体解释：
异步通信与计算重叠：这张图的关键点在于展示如何通过异步通信与计算的重叠来优化多设备环境下的计算过程。AllGather 操作的通信开销被完全隐藏在计算过程中，使得通信对计算速度的影响最小化。

分布式计算：每个设备只处理部分数据，并在每个计算阶段结束后通过 AllGather 操作将结果同步。这种方式可以在大规模并行计算中有效利用计算资源，减少设备之间的等待时间。

