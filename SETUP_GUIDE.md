# 量子CDF(2,2)小波变换 - 安装和运行指南

## 环境要求

- Python 3.8+
- 推荐使用Anaconda或虚拟环境

## 安装方法

### 方法1：快速测试（推荐）

如果您只想测试核心功能，无需安装任何外部库：

```bash
python3 pure_python_cdf_demo.py
```

### 方法2：完整功能安装

#### 对于新版本Qiskit (>=1.0)

```bash
# 创建虚拟环境（推荐）
conda create -n quantum_cdf python=3.9
conda activate quantum_cdf

# 安装依赖
pip install qiskit qiskit-aer numpy matplotlib
```

#### 对于旧版本Qiskit (<1.0)

```bash
# 创建虚拟环境（推荐）
conda create -n quantum_cdf python=3.9
conda activate quantum_cdf

# 安装依赖
pip install qiskit[all] numpy matplotlib
```

#### 完整安装（包含图像处理）

```bash
pip install -r requirements.txt
```

## 运行程序

### 1. 基础测试

检查环境和依赖：

```bash
python3 test_quantum_blocks.py
```

### 2. 纯Python演示

运行不依赖外部库的完整演示：

```bash
python3 pure_python_cdf_demo.py
```

### 3. 量子电路测试

如果Qiskit安装成功：

```bash
python3 quantum_block_circuits.py
```

### 4. 完整演示

如果所有依赖都安装成功：

```bash
python3 quantum_cdf_demo.py
```

## 故障排除

### 常见问题1：Qiskit导入错误

**错误信息：**
```
ImportError: cannot import name 'Aer' from 'qiskit'
```

**解决方案：**
```bash
# 对于Qiskit 1.0+
pip install qiskit-aer

# 或者使用旧版本
pip install qiskit==0.45.0 qiskit-aer==0.13.0
```

### 常见问题2：matplotlib显示问题

**错误信息：**
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**解决方案：**
- 图像将保存为PNG文件而不是显示
- 或者安装GUI后端：`pip install PyQt5`

### 常见问题3：缺少NumPy

**错误信息：**
```
ModuleNotFoundError: No module named 'numpy'
```

**解决方案：**
```bash
pip install numpy
```

## 程序功能说明

### 1. `pure_python_cdf_demo.py`
- ✅ 无外部依赖
- ✅ 完整的CDF(2,2)小波变换实现
- ✅ 图像块处理
- ✅ 详细的步骤输出

### 2. `quantum_block_circuits.py`
- ⚠️ 需要Qiskit
- ✅ 量子电路设计
- ✅ Split、Predict、Update三个量子块
- ✅ 电路可视化

### 3. `quantum_cdf_demo.py`
- ⚠️ 需要完整依赖
- ✅ 量子模拟
- ✅ 结果对比
- ✅ 图像处理

### 4. `test_quantum_blocks.py`
- ✅ 环境检测
- ✅ 功能测试
- ✅ 兼容性检查

## 输出结果

运行成功后，您将看到：

1. **算法步骤详解**：每个Split、Predict、Update步骤的详细计算
2. **数值结果**：近似系数和详细系数
3. **电路信息**：量子比特数、电路深度、门数量
4. **可视化图像**：保存为PNG文件（如果matplotlib可用）

## 验证结果

程序会处理论文中的4x4测试图像：
```
[1, 3, 2, 1]
[2, 6, 6, 6]
[5, 2, 7, 4]
[3, 1, 7, 2]
```

分割为4个2x2块，每个块应用CDF(2,2)变换，您可以验证结果是否符合预期。

## 技术支持

如果遇到问题：

1. 首先运行 `python3 test_quantum_blocks.py` 检查环境
2. 使用 `python3 pure_python_cdf_demo.py` 验证核心算法
3. 查看错误信息并参考故障排除部分
4. 确保Python版本为3.8+

## 扩展开发

要扩展此项目：

1. 修改 `bit_precision` 参数以支持更高精度
2. 在 `quantum_block_circuits.py` 中添加新的量子门操作
3. 扩展 `image_to_blocks` 函数以支持更大的图像
4. 添加量子错误校正功能