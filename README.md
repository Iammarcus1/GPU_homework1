# GPU_homework1
# CUDA Spiking CNN (SCNN) Inference Engine

2025年秋季国科大《GPU架构与编程》

这是一个基于 CUDA C++ 实现的高性能脉冲卷积神经网络（Spiking Convolutional Neural Network, SCNN）推理引擎。该项目针对 FashionMNIST 数据集，通过手写的 CUDA 核函数实现了卷积、IF（Integrate-and-Fire）神经元模型以及池化层的算子融合与批处理优化。

## 核心特性

- **高性能算子融合 (Kernel Fusion)**：
    - `Conv + IF + MaxPool`：将卷积、神经元发放与池化合并在一个 Kernel 中，显著降低显存带宽需求。
    - `Linear + IF`：融合全连接层与脉冲发放逻辑。
- **内存优化**：
    - 使用 **Pinned Memory (cudaHostAlloc)** 加速主机到设备（H2D）的数据传输。
    - 显存池复用策略，避免推理循环中的重复内存分配。
- **全流程 GPU 加速**：从输入编码到最终 Logits 计算全在 GPU 上完成。

## 环境依赖

- **操作系统**: Linux (推荐)
- **硬件**: NVIDIA GPU (Compute Capability >= 7.0, e.g., V100, T4, RTX 20/30/40系列)
- **编译器**: `nvcc` (CUDA Toolkit 10.0+), 支持 C++14

## 目录结构说明

**重要**：代码中对数据路径有硬编码要求 (`dir + "/../../.." + "/data/..."`)。请务必严格按照以下目录结构存放文件，否则程序将无法找到数据。

```text
project_root/
├── data/
│   └── FashionMNIST/
│       └── raw/
│           ├── t10k-images-idx3-ubyte  # 测试集图像
│           └── t10k-labels-idx1-ubyte  # 测试集标签
├── model/                     # <--- 运行程序时，将此文件夹路径作为参数传入
│   ├── conv1.weight.txt
│   ├── conv1.bias.txt
│   ├── ...
│   ├── fc3.weight.txt
│   └── fc3.bias.txt
└── inference.cu               # 源代码
