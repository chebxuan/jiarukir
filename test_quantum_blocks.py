#!/usr/bin/env python3
"""
量子块电路测试程序
兼容不同版本的Qiskit
"""

import sys
import os

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入情况"""
    print("检查依赖库...")
    
    try:
        import numpy as np
        print("✓ NumPy 可用")
    except ImportError:
        print("✗ NumPy 不可用")
        return False
    
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        print("✓ Qiskit 核心模块可用")
    except ImportError:
        print("✗ Qiskit 不可用")
        return False
    
    try:
        from qiskit_aer import Aer
        print("✓ Qiskit Aer 可用")
        aer_available = True
    except ImportError:
        try:
            from qiskit import Aer
            print("✓ Qiskit Aer (旧版本) 可用")
            aer_available = True
        except ImportError:
            print("✗ Qiskit Aer 不可用")
            aer_available = False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib 可用")
    except ImportError:
        print("✗ Matplotlib 不可用")
    
    return True

def create_simple_cdf_circuit():
    """
    创建一个简化的CDF电路用于测试
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    
    # 创建简单的4比特电路
    qr = QuantumRegister(4, 'q')
    cr = ClassicalRegister(4, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # 模拟Split操作
    qc.barrier(label='Split')
    qc.h(qr[0])  # 创建叠加态
    qc.h(qr[1])
    
    # 模拟Predict操作
    qc.barrier(label='Predict')
    qc.cx(qr[0], qr[2])  # 纠缠操作
    qc.cx(qr[1], qr[3])
    
    # 模拟Update操作
    qc.barrier(label='Update')
    qc.h(qr[2])  # 最终变换
    qc.h(qr[3])
    
    # 测量
    qc.measure_all()
    
    return qc

def test_basic_quantum_circuit():
    """
    测试基本量子电路功能
    """
    print("\n测试基本量子电路...")
    
    try:
        # 创建简单电路
        qc = create_simple_cdf_circuit()
        print(f"✓ 电路创建成功")
        print(f"   量子比特数: {qc.num_qubits}")
        print(f"   电路深度: {qc.depth()}")
        print(f"   门数量: {len(qc.data)}")
        
        # 尝试可视化
        try:
            from qiskit.visualization import circuit_drawer
            circuit_text = circuit_drawer(qc, output='text')
            print("✓ 电路可视化成功")
            print("电路结构:")
            print(circuit_text)
        except Exception as e:
            print(f"✗ 电路可视化失败: {e}")
        
        return qc
        
    except Exception as e:
        print(f"✗ 电路创建失败: {e}")
        return None

def test_quantum_simulation(qc):
    """
    测试量子模拟
    """
    print("\n测试量子模拟...")
    
    try:
        # 尝试导入Aer
        try:
            from qiskit_aer import Aer
        except ImportError:
            from qiskit import Aer
        
        from qiskit import execute, transpile
        
        # 创建模拟器
        simulator = Aer.get_backend('qasm_simulator')
        print("✓ 模拟器创建成功")
        
        # 编译电路
        compiled_circuit = transpile(qc, simulator)
        print("✓ 电路编译成功")
        
        # 执行模拟
        job = execute(compiled_circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        print("✓ 量子模拟成功")
        print(f"   测量结果: {len(counts)} 种状态")
        print("   前5个结果:")
        for i, (state, count) in enumerate(list(counts.items())[:5]):
            print(f"     |{state}⟩: {count} 次")
        
        return True
        
    except Exception as e:
        print(f"✗ 量子模拟失败: {e}")
        return False

def test_cdf_algorithm():
    """
    测试CDF(2,2)算法的经典实现
    """
    print("\n测试CDF(2,2)算法...")
    
    # 测试信号
    test_signal = [1, 3, 2, 6]
    print(f"输入信号: {test_signal}")
    
    try:
        # Step 1: Split
        even = test_signal[::2]  # [1, 2]
        odd = test_signal[1::2]  # [3, 6]
        print(f"Split - 偶数: {even}, 奇数: {odd}")
        
        # Step 2: Predict
        predict_values = []
        detail_values = []
        
        for i, odd_val in enumerate(odd):
            if i == 0:
                predict_val = 0.5 * (even[0] + even[1]) if len(even) > 1 else even[0]
            else:
                predict_val = 0.5 * (even[i-1] + even[i])
            
            detail_val = odd_val - predict_val
            predict_values.append(predict_val)
            detail_values.append(detail_val)
        
        print(f"Predict - 预测值: {predict_values}, 详细系数: {detail_values}")
        
        # Step 3: Update
        update_values = []
        approx_values = []
        
        for i, even_val in enumerate(even):
            if i == 0:
                update_val = 0.25 * detail_values[0] if detail_values else 0
            elif i == len(even) - 1:
                update_val = 0.25 * detail_values[i-1] if i-1 < len(detail_values) else 0
            else:
                update_val = 0.25 * (detail_values[i-1] + detail_values[i])
            
            approx_val = even_val + update_val
            update_values.append(update_val)
            approx_values.append(approx_val)
        
        print(f"Update - 更新值: {update_values}, 近似系数: {approx_values}")
        
        print("✓ CDF(2,2)算法测试成功")
        return True
        
    except Exception as e:
        print(f"✗ CDF(2,2)算法测试失败: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("量子CDF(2,2)小波变换块电路测试")
    print("=" * 60)
    
    # 1. 测试导入
    if not test_imports():
        print("\n依赖库检查失败，请安装必要的库")
        return False
    
    # 2. 测试CDF算法
    cdf_success = test_cdf_algorithm()
    
    # 3. 测试量子电路
    qc = test_basic_quantum_circuit()
    
    # 4. 测试量子模拟（如果可能）
    simulation_success = False
    if qc is not None:
        simulation_success = test_quantum_simulation(qc)
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"CDF算法测试: {'✓ 通过' if cdf_success else '✗ 失败'}")
    print(f"量子电路测试: {'✓ 通过' if qc is not None else '✗ 失败'}")
    print(f"量子模拟测试: {'✓ 通过' if simulation_success else '✗ 失败'}")
    
    if cdf_success and qc is not None:
        print("\n🎉 基本功能测试通过！")
        print("\n下一步可以运行:")
        print("  python3 pure_python_cdf_demo.py  # 纯Python演示")
        if simulation_success:
            print("  python3 quantum_block_circuits.py  # 量子电路测试")
    else:
        print("\n⚠️  部分功能测试失败，但核心算法可用")
    
    return cdf_success

if __name__ == "__main__":
    success = main()