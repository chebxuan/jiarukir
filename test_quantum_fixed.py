#!/usr/bin/env python3
"""
测试修复后的Qiskit导入和基本功能
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile

def test_basic_imports():
    """测试基本导入"""
    print("✓ 所有导入成功!")
    return True

def test_quantum_circuit():
    """测试量子电路创建和执行"""
    try:
        # 创建简单的量子电路
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        print("✓ 量子电路创建成功!")
        return True
    except Exception as e:
        print(f"✗ 量子电路创建失败: {e}")
        return False

def test_aer_backend():
    """测试Aer模拟器"""
    try:
        backend = Aer.get_backend('qasm_simulator')
        print(f"✓ Aer模拟器获取成功: {backend.name}")
        return True
    except Exception as e:
        print(f"✗ Aer模拟器获取失败: {e}")
        return False

def test_transpile_and_run():
    """测试新的transpile和run模式"""
    try:
        # 创建测试电路
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        
        # 使用新的执行方式
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(qc, backend)
        job = backend.run(transpiled_qc, shots=100)
        result = job.result()
        counts = result.get_counts()
        
        print(f"✓ 量子电路执行成功! 结果: {counts}")
        return True
    except Exception as e:
        print(f"✗ 量子电路执行失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Qiskit 2.1.1 兼容性测试")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_quantum_circuit,
        test_aer_backend,
        test_transpile_and_run
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过! 您的环境已正确配置")
    else:
        print("⚠️  部分测试失败，请检查配置")
    
    print("=" * 50)

if __name__ == "__main__":
    main()