# ⚡ Tensor-cpp: Professional C++ Tensor Library from Scratch

**Lightweight, Modular, and Trainable Deep Learning Tensor Engine with Reverse-Mode Automatic Differentiation (Inspired by TensorFlow & PyTorch)**

[![Language: C++17](https://img.shields.io/badge/Language-C%2B%2B17-00599C.svg?style=flat-square&logo=c%2B%2B)](https://en.cppreference.com/w/cpp/17)
[![Architecture: Handle--Body](https://img.shields.io/badge/Architecture-Handle--Body-8A2BE2.svg?style=flat-square)]()
[![Autodiff: Reverse--Mode](https://img.shields.io/badge/Autodiff-Reverse--Mode-2E8B57.svg?style=flat-square)]()
[![OS: Windows | Linux | macOS](https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-0078D4.svg?style=flat-square&logo=windows)]()

---

🌐 [English Documentation](#-english-documentation) | 🇨🇳 [中文说明 (Mandarin)](#-中文说明-mandarin) | 🇮🇩 [Bahasa Indonesia](#-bahasa-indonesia)

---

## 🚀 English Documentation

### 📌 Overview
**Tensor-cpp** is a professional, industrial-grade C++17 library built from scratch to handle multi-dimensional numerical data and dynamic computation graphs. Designed with the architecture of modern deep learning frameworks (like **TensorFlow** and **PyTorch**), it supports automatic differentiation (Autodiff), matrix inversions, and neural network training steps.

### 🏛️ System Architecture

The project follows clean software engineering principles and an extreme modular design:

```
+-----------------------------------------------------------------+
|                         Tensor (Handle)                         |
|  - Manages shape, strides, and shared ownership                 |
|  - Operator Overloads (+, -, *, /, -)                           |
+-----------------------------------------------------------------+
                                 |
                                 | std::shared_ptr<TensorImpl>
                                 v
+-----------------------------------------------------------------+
|                       TensorImpl (Body)                         |
|  - std::vector<double> data  |  std::vector<double> grad        |
|  - bool requires_grad        |  std::function<void()> backward  |
|  - std::vector<Tensor> parents (Computation Graph Node)         |
+-----------------------------------------------------------------+
                                 ^
                                 | Attached via weak_ptr closures
+-----------------------------------------------------------------+
|             Modular Operations (ops:: Namespace)                |
|  - File-per-Operation: relu.cpp, sigmoid.cpp, matmul.cpp, etc.  |
|  - Forward computation + Backward gradient propagation rules    |
+-----------------------------------------------------------------+
```

#### 1. Handle-Body Idiom (`Tensor` & `TensorImpl`)
To allow seamless sharing of tensors across computation nodes without unnecessary deep memory copies, the library uses the **Handle-Body idiom**:
- `Tensor` acts as a lightweight pointer handle.
- `TensorImpl` stores the actual multi-dimensional `data`, `grad` buffer, dimensions (`shape`), and graph dependencies (`parents`).

#### 2. Reverse-Mode Automatic Differentiation (Autodiff Engine)
When operations like `ops::matmul(X, W)` or `ops::sin(x)` are executed, the framework builds a **Dynamic Computation Graph**:
- Each output tensor stores a `backward_fn` lambda closure containing the analytical chain-rule formula.
- To prevent memory leaks from cyclic references, closures hold weak pointers (`std::weak_ptr<TensorImpl>`).
- Calling `.backward()` triggers a **Topological Sort (Depth-First Search)** that propagates gradient flow (`dL/dx`) backwards from the loss scalar to all trainable weights.

#### 3. Extreme Modularity (File-per-Operation)
Every single mathematical operation and neural network activation function lives in its own dedicated `.hpp` and `.cpp` file inside `include/ops/` and `src/ops/`. A unified aggregator header `include/ops/all_ops.hpp` bundles them cleanly for end users.

---

### 🧮 Available Modules & Operations

| Category | Available Operations | Backward Gradient Support |
| :--- | :--- | :---: |
| **Basic Algebra** | `add`, `sub`, `mul`, `div`, `neg`, `pow`, `exp`, `log` | ✅ Trainable (Full Autodiff) |
| **Trigonometry** | `sin`, `cos`, `tan`, `tanh` | ✅ Trainable (Full Autodiff) |
| **Activations** | `relu`, `sigmoid`, `softmax` | ✅ Trainable (Full Autodiff) |
| **Linear Algebra** | `matmul`, `dot`, `transpose`, `inverse` | ✅ Trainable (Full Autodiff) |
| **Reductions** | `sum`, `mean` | ✅ Trainable (Full Autodiff) |

---

### 💻 How to Build and Run (Multi-OS Guide)

#### 🪟 1. Windows
**Option A: Microsoft Visual Studio (Recommended - MSVC `cl.exe`)**
Open **Developer Command Prompt for VS** or **x64 Native Tools Command Prompt**:
```cmd
cl /EHsc /std:c++17 main.cpp src\Tensor.cpp src\ops\*.cpp /Fe:main.exe
main.exe
```

**Option B: MinGW / GCC via PowerShell or CMD**
Ensure MinGW (`g++`) is added to your Windows Environment `PATH`:
```bash
g++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main.exe
.\main.exe
```

#### 🐧 2. Linux (Ubuntu / Debian / Fedora / Arch)
Ensure `build-essential` or GCC/Clang is installed (`sudo apt install build-essential`):
```bash
g++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main
./main
```

#### 🍎 3. macOS (Apple Silicon M1/M2/M3 & Intel)
Using Apple Clang via Xcode Command Line Tools (`xcode-select --install`):
```bash
clang++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main
./main
```

---

## 🇨🇳 中文说明 (Mandarin)

### 📌 项目简介
**Tensor-cpp** 是一个完全从零用 **C++17** 构建的工业级多维张量与深度学习核心库。该库的设计深度参考了 **TensorFlow** 和 **PyTorch** 的底层架构，不仅具备处理高维数据矩阵的能力，还全面实现了反向传播自动微分（Autodiff）、高阶线性代数运算（如矩阵求逆）以及神经网络训练迭代更新。

### 🏛️ 架构设计解析

本项目采用了极度模块化与高标准的软件工程设计：

1. **句柄-实体模式 (Handle-Body Idiom)**：
   - 为避免在构建计算图时产生昂贵的深拷贝，核心类 `Tensor` 仅作为轻量级句柄（Handle）。
   - 真实的数值数据（`data`）、梯度缓存（`grad`）、维度形状（`shape`）及父节点依赖，均封装在由 `std::shared_ptr<TensorImpl>` 管理的实体对象中。

2. **动态反向传播自动微分 (Reverse-Mode Autodiff)**：
   - 每次执行数学运算时（例如矩阵乘法或激活函数），系统会自动构建**动态计算图**。
   - 运算输出张量会绑定一个 `backward_fn` 闭包函数，记录当前步骤的链式法则求导公式。闭包内部采用 `std::weak_ptr` 彻底避免了循环引用导致的内存泄露。
   - 调用 `.backward()` 时，引擎通过**深度优先拓扑排序 (Topological Sort)**，精准地将梯度从损失函数（Loss）反向传播至所有标记为 `requires_grad=true` 的权重参数。

3. **极度模块化函数设计 (File-per-Operation)**：
   - 告别臃肿的单体代码，所有数学运算与激活函数（如 `relu`, `sigmoid`, `matmul`, `softmax` 等）均拥有完全独立的 `.hpp` 头文件与 `.cpp` 实现文件（位于 `include/ops/` 和 `src/ops/`）。
   - 用户只需引入聚合头文件 `include/ops/all_ops.hpp` 即可便捷调用 `ops::` 命名空间下的全部功能。

---

### 💻 多操作系统编译与运行指南 (Multi-OS Guide)

#### 🪟 1. Windows 系统
**方式 A：使用 Microsoft Visual Studio (推荐 MSVC)**
打开 **Developer Command Prompt for VS** 终端：
```cmd
cl /EHsc /std:c++17 main.cpp src\Tensor.cpp src\ops\*.cpp /Fe:main.exe
main.exe
```

**方式 B：使用 MinGW / GCC (PowerShell 或 CMD)**
```bash
g++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main.exe
.\main.exe
```

#### 🐧 2. Linux 系统 (Ubuntu / Debian / CentOS)
确保已安装 `build-essential` 编译工具包：
```bash
g++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main
./main
```

#### 🍎 3. macOS 系统 (Apple Silicon 芯片 & Intel)
使用 Xcode 命令行工具提供的 Apple Clang (`xcode-select --install`)：
```bash
clang++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main
./main
```

---

## 🇮🇩 Bahasa Indonesia

### 📌 Ringkasan Proyek
**Tensor-cpp** adalah library pustaka tensor C++17 standar industri yang dibuat dari nol (*from scratch*) untuk pemrosesan komputasi numerik dan deep learning. Mengadopsi desain arsitektur modern layaknya **TensorFlow** dan **PyTorch**, library ini mendukung Automatic Differentiation (Autodiff), aljabar linier kompleks, dan optimasi langkah training jaringan saraf tiruan.

### 🏛️ Sorotan Arsitektur
1. **Pemisahan Handle-Body**: Menggunakan `std::shared_ptr<TensorImpl>` agar perpindahan data antar node komputasi berlangsung sangat cepat tanpa salinan memori yang tidak perlu.
2. **Mesin Autodiff Topologis**: Menggunakan penelusuran graf DFS (*Topological Sort*) untuk menghitung gradien turunan secara akurat. Penggunaan *weak pointer* menjamin bebas kebocoran memori.
3. **Modularitas Ekstrem (File-per-Operation)**: Setiap fungsi matematika (`sin`, `cos`, `relu`, `matmul`, dll.) dipisahkan ke dalam file `.hpp` dan `.cpp` tersendiri di dalam direktori `ops/`, menjadikan struktur kode sangat rapi, profesional, dan mudah dikembangkan.

---

### 💻 Panduan Kompilasi & Eksekusi Multi-OS

#### 🪟 1. Windows
**Opsi A: Microsoft Visual Studio (Rekomendasi - MSVC `cl.exe`)**
Buka terminal **Developer Command Prompt for VS**:
```cmd
cl /EHsc /std:c++17 main.cpp src\Tensor.cpp src\ops\*.cpp /Fe:main.exe
main.exe
```

**Opsi B: MinGW / GCC di PowerShell atau CMD**
Pastikan MinGW sudah ditambahkan ke `PATH` Windows:
```bash
g++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main.exe
.\main.exe
```

#### 🐧 2. Linux (Ubuntu / Debian / Fedora / Arch)
Pastikan compiler GCC/Clang sudah terinstall (`sudo apt install build-essential`):
```bash
g++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main
./main
```

#### 🍎 3. macOS (Apple Silicon M1/M2/M3 & Intel)
Menggunakan compiler bawaan Apple Clang via Xcode Command Line Tools:
```bash
clang++ -std=c++17 main.cpp src/Tensor.cpp src/ops/*.cpp -o main
./main
```

---
*Created and refactored with industry-best C++ practices for high-performance numerical computing.*
