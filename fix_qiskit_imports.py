#!/usr/bin/env python3
"""
Qiskit导入问题修复脚本
针对不同版本的Qiskit提供兼容性解决方案
"""

def check_qiskit_version():
    """检查Qiskit版本并提供修复建议"""
    print("检查Qiskit安装状态...")
    
    try:
        import qiskit
        print(f"✓ Qiskit已安装，版本: {qiskit.__version__}")
        
        # 检查Aer导入
        aer_status = check_aer_import()
        
        if aer_status:
            print("✓ Qiskit环境配置正确")
            return True
        else:
            print("⚠️ Aer模块需要修复")
            return False
            
    except ImportError:
        print("✗ Qiskit未安装")
        print_installation_guide()
        return False

def check_aer_import():
    """检查Aer导入方式"""
    
    # 方法1：新版本Qiskit
    try:
        from qiskit_aer import Aer
        print("✓ Qiskit Aer (新版本) 导入成功")
        return True
    except ImportError:
        pass
    
    # 方法2：旧版本Qiskit
    try:
        from qiskit import Aer
        print("✓ Qiskit Aer (旧版本) 导入成功")
        return True
    except ImportError:
        pass
    
    print("✗ Aer模块导入失败")
    return False

def print_installation_guide():
    """打印安装指南"""
    print("\n" + "="*60)
    print("Qiskit安装指南")
    print("="*60)
    
    print("""
针对您的错误 "cannot import name 'Aer' from 'qiskit'"，请尝试以下解决方案：

方案1：安装qiskit-aer（推荐）
```
pip install qiskit-aer
```

方案2：降级到兼容版本
```
pip install qiskit==0.45.0 qiskit-aer==0.13.0
```

方案3：完全重新安装
```
pip uninstall qiskit qiskit-aer
pip install qiskit qiskit-aer
```

方案4：使用conda安装
```
conda install -c conda-forge qiskit qiskit-aer
```
""")

def create_fixed_quantum_circuit():
    """创建修复后的量子电路示例"""
    print("\n创建兼容的量子电路...")
    
    try:
        from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
        
        # 创建简单的CDF电路
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # Split模拟
        qc.barrier(label='Split')
        qc.h(qr[0])
        qc.h(qr[1])
        
        # Predict模拟
        qc.barrier(label='Predict')
        qc.cx(qr[0], qr[2])
        qc.cx(qr[1], qr[3])
        
        # Update模拟
        qc.barrier(label='Update')
        qc.h(qr[2])
        qc.h(qr[3])
        
        qc.measure_all()
        
        print("✓ 量子电路创建成功")
        print(f"   量子比特数: {qc.num_qubits}")
        print(f"   电路深度: {qc.depth()}")
        
        # 尝试模拟
        return test_simulation(qc)
        
    except Exception as e:
        print(f"✗ 量子电路创建失败: {e}")
        return False

def test_simulation(qc):
    """测试量子模拟"""
    print("\n测试量子模拟...")
    
    # 尝试不同的Aer导入方式
    simulator = None
    
    try:
        from qiskit_aer import Aer
        simulator = Aer.get_backend('qasm_simulator')
        print("✓ 使用qiskit_aer.Aer")
    except ImportError:
        try:
            from qiskit import Aer
            simulator = Aer.get_backend('qasm_simulator')
            print("✓ 使用qiskit.Aer")
        except ImportError:
            print("✗ 无法导入Aer模拟器")
            return False
    
    if simulator is None:
        return False
    
    try:
        from qiskit import execute, transpile
        
        # 编译和执行
        compiled_circuit = transpile(qc, simulator)
        job = execute(compiled_circuit, simulator, shots=100)
        result = job.result()
        counts = result.get_counts()
        
        print("✓ 量子模拟成功")
        print(f"   测量结果数量: {len(counts)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 量子模拟失败: {e}")
        return False

def create_compatibility_wrapper():
    """创建兼容性包装器"""
    wrapper_code = '''
# Qiskit兼容性包装器
# 将此代码添加到您的quantum_block_circuits.py文件开头

try:
    from qiskit import Aer, execute, transpile
except ImportError:
    try:
        from qiskit_aer import Aer
        from qiskit import execute, transpile
    except ImportError:
        print("警告: Qiskit Aer不可用，某些功能可能无法使用")
        Aer = None
        
def safe_get_backend(backend_name='qasm_simulator'):
    """安全获取后端"""
    if Aer is None:
        raise ImportError("Aer模拟器不可用")
    return Aer.get_backend(backend_name)

def safe_execute(circuit, backend, shots=1024):
    """安全执行量子电路"""
    if Aer is None:
        raise ImportError("Aer模拟器不可用")
    compiled = transpile(circuit, backend)
    job = execute(compiled, backend, shots=shots)
    return job.result()
'''
    
    with open('qiskit_compatibility.py', 'w', encoding='utf-8') as f:
        f.write(wrapper_code)
    
    print("✓ 创建了兼容性包装器: qiskit_compatibility.py")

def main():
    """主函数"""
    print("="*60)
    print("Qiskit导入问题诊断和修复")
    print("="*60)
    
    # 1. 检查Qiskit版本
    qiskit_ok = check_qiskit_version()
    
    # 2. 测试量子电路
    if qiskit_ok:
        circuit_ok = create_fixed_quantum_circuit()
        
        if circuit_ok:
            print("\n🎉 Qiskit环境配置正确！")
            print("\n现在您可以运行:")
            print("  python3 quantum_block_circuits.py")
        else:
            print("\n⚠️ 量子模拟存在问题，但电路创建正常")
    else:
        print("\n❌ 需要修复Qiskit安装")
    
    # 3. 创建兼容性包装器
    create_compatibility_wrapper()
    
    print(f"\n📋 总结:")
    print(f"- Qiskit基础: {'✓' if qiskit_ok else '✗'}")
    print(f"- 量子电路: {'✓' if qiskit_ok else '✗'}")
    print(f"- 兼容包装: ✓")
    
    print(f"\n🔧 如果问题仍然存在，请:")
    print(f"1. 运行: pip install qiskit-aer")
    print(f"2. 或使用纯Python版本: python3 standalone_cdf_test.py")

if __name__ == "__main__":
    main()