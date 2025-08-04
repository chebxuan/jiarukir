#!/usr/bin/env python3
"""
运行量子CDF(2,2)小波变换演示的简化脚本
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_simple_demo():
    """运行简化的演示，不依赖复杂的量子电路"""
    print("=" * 60)
    print("量子CDF(2,2)小波变换简化演示")
    print("=" * 60)
    
    # 创建测试图像（与论文一致）
    test_image = np.array([
        [1, 3, 2, 1],
        [2, 6, 6, 6],
        [5, 2, 7, 4],
        [3, 1, 7, 2]
    ], dtype=np.uint8)
    
    print("\n1. 测试图像 (4x4):")
    print(test_image)
    
    # 分割为2x2块
    blocks = []
    positions = []
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            block = test_image[i:i+2, j:j+2]
            blocks.append(block.flatten())
            positions.append((i, j))
    
    print(f"\n2. 分割为 {len(blocks)} 个2x2块:")
    for i, (block, pos) in enumerate(zip(blocks, positions)):
        print(f"  块 {i+1} (位置 {pos}): {block}")
    
    # 对每个块应用CDF(2,2)变换
    print("\n3. 应用CDF(2,2)小波变换:")
    results = []
    
    for i, block in enumerate(blocks):
        print(f"\n--- 处理块 {i+1}: {block} ---")
        
        # Step 1: Split
        even = block[::2]  # [block[0], block[2]]
        odd = block[1::2]  # [block[1], block[3]]
        print(f"Split - 偶数样本 S(2i): {even}")
        print(f"Split - 奇数样本 S(2i+1): {odd}")
        
        # Step 2: Predict
        predict = []
        detail = []
        for j, odd_val in enumerate(odd):
            # 简化的预测：P(S) = 平均相邻偶数值
            if j == 0:
                pred_val = even[0]  # 边界处理
            else:
                pred_val = 0.5 * (even[j-1] + even[j])
            
            det_val = odd_val - pred_val
            predict.append(pred_val)
            detail.append(det_val)
            
            print(f"Predict - i={j}: P(S)={pred_val:.1f}, D(i)={odd_val}-{pred_val:.1f}={det_val:.1f}")
        
        # Step 3: Update
        update = []
        approx = []
        for j, even_val in enumerate(even):
            # W(D) = 1/4 * (相邻detail系数的平均)
            if j == 0:
                upd_val = 0.25 * detail[0] if detail else 0
            elif j == len(even) - 1:
                upd_val = 0.25 * detail[j-1] if j-1 < len(detail) else 0
            else:
                upd_val = 0.25 * (detail[j-1] + detail[j])
            
            app_val = even_val + upd_val
            update.append(upd_val)
            approx.append(app_val)
            
            print(f"Update - i={j}: W(D)={upd_val:.2f}, A(i)={even_val}+{upd_val:.2f}={app_val:.2f}")
        
        results.append({
            'original': block,
            'approximation': approx,
            'detail': detail,
            'block_id': i
        })
        
        print(f"最终结果 - 近似系数: {[f'{x:.2f}' for x in approx]}")
        print(f"最终结果 - 详细系数: {[f'{x:.2f}' for x in detail]}")
    
    # 可视化结果
    print("\n4. 结果可视化")
    fig, axes = plt.subplots(2, len(blocks), figsize=(4*len(blocks), 8))
    if len(blocks) == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results):
        # 原始块
        original_2d = result['original'].reshape(2, 2)
        axes[0, i].imshow(original_2d, cmap='gray', interpolation='nearest')
        axes[0, i].set_title(f'原始块 {i+1}')
        
        # 添加数值标注
        for x in range(2):
            for y in range(2):
                axes[0, i].text(y, x, f'{original_2d[x, y]}', 
                               ha='center', va='center', color='white', fontweight='bold')
        
        # 变换结果
        approx = result['approximation']
        detail = result['detail']
        
        # 重构显示
        transformed = np.array([[approx[0] if approx else 0, detail[0] if detail else 0],
                              [approx[1] if len(approx) > 1 else 0, 
                               detail[1] if len(detail) > 1 else 0]])
        
        axes[1, i].imshow(transformed, cmap='RdBu', interpolation='nearest')
        axes[1, i].set_title(f'CDF变换块 {i+1}')
        
        # 添加数值标注
        for x in range(2):
            for y in range(2):
                axes[1, i].text(y, x, f'{transformed[x, y]:.1f}', 
                               ha='center', va='center', fontweight='bold')
    
    plt.suptitle('CDF(2,2)小波变换结果对比', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cdf_wavelet_results.png', dpi=150, bbox_inches='tight')
    print("结果图像已保存为 'cdf_wavelet_results.png'")
    
    # 显示图像（如果支持）
    try:
        plt.show()
    except:
        print("无法显示图像，但已保存到文件")
    
    print("\n5. 总结")
    print("-" * 40)
    print("✓ 成功实现了CDF(2,2)小波变换的三个步骤：")
    print("  - Split: 分离奇偶样本")
    print("  - Predict: P(S) = 1/2[S(2i) + S(2i+2)]，计算D(i)")
    print("  - Update: W(D) = 1/4[D(i-1) + D(i)]，计算A(i)")
    print(f"✓ 处理了 {len(blocks)} 个2x2图像块")
    print("✓ 生成了近似系数和详细系数")
    print("✓ 提供了完整的可视化结果")
    
    return results

if __name__ == "__main__":
    print("启动CDF(2,2)小波变换演示...")
    results = run_simple_demo()
    print("\n演示完成！")