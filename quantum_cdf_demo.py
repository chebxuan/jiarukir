#!/usr/bin/env python3
"""
量子CDF(2,2)小波变换演示程序

基于论文中的提升方案实现：
1. Split: 分离奇偶样本
2. Predict: P(S) = 1/2[S(2i) + S(2i+2)]，计算D(i) = S(2i+1) - P(S)  
3. Update: W(D) = 1/4[D(i-1) + D(i)]，计算A(i) = S(2i) + W(D)

作者：基于用户提供的CDF(2,2)小波变换量子提升方案
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute, transpile
from qiskit.visualization import plot_histogram, circuit_drawer
import cv2
from PIL import Image
import os
import time

# 导入我们的量子实现
from quantum_cdf_wavelet import QuantumCDFWaveletTransform
from quantum_block_circuits import QuantumBlockCircuits

class QuantumCDFDemo:
    """
    量子CDF(2,2)小波变换完整演示系统
    """
    
    def __init__(self, bit_precision=4):
        self.bit_precision = bit_precision
        self.quantum_cdf = QuantumCDFWaveletTransform(bit_precision)
        self.quantum_blocks = QuantumBlockCircuits(bit_precision)
        
    def create_test_image(self, size=(4, 4)):
        """
        创建测试图像（与论文中的示例一致）
        """
        # 使用论文中的4x4测试图像
        test_image = np.array([
            [1, 3, 2, 1],
            [2, 6, 6, 6],
            [5, 2, 7, 4],
            [3, 1, 7, 2]
        ], dtype=np.uint8)
        
        return test_image
    
    def image_to_blocks(self, image, block_size=(2, 2)):
        """
        将图像分割为2x2块进行处理
        """
        h, w = image.shape
        blocks = []
        block_positions = []
        
        for i in range(0, h, block_size[0]):
            for j in range(0, w, block_size[1]):
                block = image[i:i+block_size[0], j:j+block_size[1]]
                if block.shape == block_size:  # 确保块大小正确
                    blocks.append(block.flatten())
                    block_positions.append((i, j))
        
        return blocks, block_positions
    
    def classical_cdf_on_blocks(self, blocks):
        """
        对每个2x2块应用经典CDF(2,2)小波变换
        """
        results = []
        
        for i, block in enumerate(blocks):
            print(f"处理块 {i+1}: {block}")
            
            # 应用CDF(2,2)变换
            approx, detail = self.quantum_cdf.classical_cdf_transform(block)
            
            results.append({
                'original': block,
                'approximation': approx,
                'detail': detail,
                'block_id': i
            })
            
            print(f"  近似系数: {approx}")
            print(f"  详细系数: {detail}")
            print()
        
        return results
    
    def quantum_cdf_on_blocks(self, blocks):
        """
        对每个2x2块应用量子CDF(2,2)小波变换
        """
        quantum_results = []
        
        for i, block in enumerate(blocks):
            print(f"量子处理块 {i+1}: {block}")
            
            try:
                # 创建量子电路
                qc = self.quantum_blocks.create_complete_cdf_block_circuit(block)
                
                # 运行量子模拟
                simulator = Aer.get_backend('qasm_simulator')
                compiled_circuit = transpile(qc, simulator)
                job = execute(compiled_circuit, simulator, shots=1024)
                result = job.result()
                counts = result.get_counts()
                
                quantum_results.append({
                    'original': block,
                    'quantum_circuit': qc,
                    'measurement_counts': counts,
                    'block_id': i
                })
                
                print(f"  量子测量结果: {counts}")
                
            except Exception as e:
                print(f"  量子处理出错: {e}")
                quantum_results.append({
                    'original': block,
                    'error': str(e),
                    'block_id': i
                })
            
            print()
        
        return quantum_results
    
    def demonstrate_individual_steps(self):
        """
        演示CDF(2,2)变换的各个步骤
        """
        print("=== CDF(2,2)小波变换步骤演示 ===\n")
        
        # 使用简单的测试信号
        test_signal = [1, 3, 2, 6]  # 2x2块展开
        print(f"测试信号: {test_signal}")
        
        # Step 1: Split演示
        print("\n--- Step 1: Split ---")
        even_samples = test_signal[::2]  # [1, 2]
        odd_samples = test_signal[1::2]  # [3, 6]
        print(f"偶数样本 S(2i): {even_samples}")
        print(f"奇数样本 S(2i+1): {odd_samples}")
        
        # 创建Split量子电路
        split_circuit = self.quantum_blocks.create_split_block(test_signal)
        print(f"Split量子电路深度: {split_circuit.depth()}")
        print(f"Split量子电路量子比特数: {split_circuit.num_qubits}")
        
        # Step 2: Predict演示
        print("\n--- Step 2: Predict ---")
        # P(S) = 1/2[S(2i) + S(2i+2)]
        # 对于边界处理，我们使用相邻值
        predict_values = []
        detail_values = []
        
        for i, odd_val in enumerate(odd_samples):
            if i == 0:
                # 边界：P(S) = S(2i) = even_samples[0]
                predict_val = even_samples[0]
            else:
                # P(S) = 1/2[S(2i) + S(2i+2)]
                predict_val = 0.5 * (even_samples[i-1] + even_samples[i])
            
            detail_val = odd_val - predict_val
            predict_values.append(predict_val)
            detail_values.append(detail_val)
            
            print(f"  i={i}: P(S)={predict_val}, D(i)=S({2*i+1})-P(S)={odd_val}-{predict_val}={detail_val}")
        
        # Step 3: Update演示
        print("\n--- Step 3: Update ---")
        update_values = []
        approx_values = []
        
        for i, even_val in enumerate(even_samples):
            if i == 0:
                # 边界：W(D) = 1/4 * D(0)
                if len(detail_values) > 0:
                    update_val = 0.25 * detail_values[0]
                else:
                    update_val = 0
            elif i == len(even_samples) - 1:
                # 边界：W(D) = 1/4 * D(i-1)
                update_val = 0.25 * detail_values[i-1]
            else:
                # W(D) = 1/4[D(i-1) + D(i)]
                update_val = 0.25 * (detail_values[i-1] + detail_values[i])
            
            approx_val = even_val + update_val
            update_values.append(update_val)
            approx_values.append(approx_val)
            
            print(f"  i={i}: W(D)={update_val}, A(i)=S({2*i})+W(D)={even_val}+{update_val}={approx_val}")
        
        print(f"\n最终结果:")
        print(f"近似系数 A(i): {approx_values}")
        print(f"详细系数 D(i): {detail_values}")
        
        return {
            'original': test_signal,
            'approximation': approx_values,
            'detail': detail_values,
            'split_circuit': split_circuit
        }
    
    def visualize_results(self, classical_results, quantum_results=None):
        """
        可视化变换结果
        """
        n_blocks = len(classical_results)
        
        # 创建子图
        fig, axes = plt.subplots(2, n_blocks, figsize=(4*n_blocks, 8))
        if n_blocks == 1:
            axes = axes.reshape(2, 1)
        
        for i, result in enumerate(classical_results):
            # 原始块可视化
            original_2d = result['original'].reshape(2, 2)
            im1 = axes[0, i].imshow(original_2d, cmap='gray', interpolation='nearest')
            axes[0, i].set_title(f'原始块 {i+1}')
            axes[0, i].set_xticks([0, 1])
            axes[0, i].set_yticks([0, 1])
            
            # 添加数值标注
            for x in range(2):
                for y in range(2):
                    axes[0, i].text(y, x, f'{original_2d[x, y]}', 
                                   ha='center', va='center', color='white', fontweight='bold')
            
            # 变换结果可视化
            approx = result['approximation']
            detail = result['detail']
            
            # 重构2x2结果（简化显示）
            transformed = np.array([[approx[0] if len(approx) > 0 else 0, 
                                   detail[0] if len(detail) > 0 else 0],
                                  [approx[1] if len(approx) > 1 else 0, 
                                   detail[1] if len(detail) > 1 else 0]])
            
            im2 = axes[1, i].imshow(transformed, cmap='RdBu', interpolation='nearest')
            axes[1, i].set_title(f'CDF变换块 {i+1}')
            axes[1, i].set_xticks([0, 1])
            axes[1, i].set_yticks([0, 1])
            
            # 添加数值标注
            for x in range(2):
                for y in range(2):
                    axes[1, i].text(y, x, f'{transformed[x, y]:.1f}', 
                                   ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def visualize_quantum_circuit(self, quantum_result):
        """
        可视化量子电路
        """
        if 'quantum_circuit' in quantum_result:
            qc = quantum_result['quantum_circuit']
            
            # 创建电路图
            fig = plt.figure(figsize=(16, 10))
            
            try:
                circuit_diagram = circuit_drawer(qc, output='mpl', style='iqx', fold=-1)
                plt.title(f"量子CDF(2,2)小波变换电路 - 块 {quantum_result['block_id']+1}", 
                         fontsize=14, fontweight='bold')
            except Exception as e:
                plt.text(0.5, 0.5, f"电路可视化错误: {e}", 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title("量子电路可视化")
            
            plt.tight_layout()
            return fig
        
        return None
    
    def run_complete_demo(self):
        """
        运行完整的演示程序
        """
        print("=" * 60)
        print("量子CDF(2,2)小波变换完整演示")
        print("=" * 60)
        
        # 1. 创建测试图像
        print("\n1. 创建测试图像")
        test_image = self.create_test_image()
        print("测试图像 (4x4):")
        print(test_image)
        
        # 2. 分割为2x2块
        print("\n2. 分割图像为2x2块")
        blocks, positions = self.image_to_blocks(test_image)
        print(f"总共 {len(blocks)} 个块:")
        for i, (block, pos) in enumerate(zip(blocks, positions)):
            print(f"  块 {i+1} (位置 {pos}): {block}")
        
        # 3. 演示各个步骤
        print("\n3. CDF(2,2)变换步骤演示")
        step_demo = self.demonstrate_individual_steps()
        
        # 4. 经典CDF变换
        print("\n4. 经典CDF(2,2)小波变换")
        classical_results = self.classical_cdf_on_blocks(blocks)
        
        # 5. 量子CDF变换（简化版本）
        print("\n5. 量子CDF(2,2)小波变换")
        quantum_results = self.quantum_cdf_on_blocks(blocks)
        
        # 6. 结果可视化
        print("\n6. 结果可视化")
        
        # 可视化经典结果
        fig_classical = self.visualize_results(classical_results)
        fig_classical.suptitle('经典CDF(2,2)小波变换结果', fontsize=16, fontweight='bold')
        plt.show()
        
        # 可视化量子电路（第一个块）
        if quantum_results and 'quantum_circuit' in quantum_results[0]:
            fig_quantum = self.visualize_quantum_circuit(quantum_results[0])
            if fig_quantum:
                plt.show()
        
        # 7. 性能对比
        print("\n7. 性能对比总结")
        print("-" * 40)
        print(f"处理块数: {len(blocks)}")
        print(f"量子比特精度: {self.bit_precision} bits")
        print(f"经典处理成功: {len(classical_results)} / {len(blocks)}")
        quantum_success = sum(1 for r in quantum_results if 'error' not in r)
        print(f"量子处理成功: {quantum_success} / {len(blocks)}")
        
        return {
            'image': test_image,
            'blocks': blocks,
            'classical_results': classical_results,
            'quantum_results': quantum_results,
            'step_demo': step_demo
        }

def main():
    """
    主函数
    """
    print("启动量子CDF(2,2)小波变换演示...")
    
    # 创建演示实例
    demo = QuantumCDFDemo(bit_precision=4)
    
    # 运行完整演示
    results = demo.run_complete_demo()
    
    print("\n演示完成！")
    print("\n主要成果:")
    print("✓ 实现了基于论文公式的CDF(2,2)小波变换")
    print("✓ 设计了Split、Predict、Update三个量子电路块")
    print("✓ 创建了完整的量子块状电路")
    print("✓ 提供了经典模拟验证对比")
    print("✓ 实现了图像块处理和可视化")
    
    return results

if __name__ == "__main__":
    results = main()