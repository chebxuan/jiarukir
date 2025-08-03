import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit import transpile
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit.quantum_info import Operator
import qiskit.quantum_info as qi
from scipy.ndimage import convolve

class QuantumSTransformCircuit:
    """量子S变换的量子电路实现"""
    
    def __init__(self, bit_precision: int = 8):
        self.bit_precision = bit_precision
        self.max_value = 2**bit_precision - 1
        
    def create_s_transform_circuit(self, even_val: int, odd_val: int):
        """创建量子S变换电路"""
        # 创建量子寄存器
        qr_even = QuantumRegister(self.bit_precision, 'even')
        qr_odd = QuantumRegister(self.bit_precision, 'odd')
        cr_even = ClassicalRegister(self.bit_precision, 'cr_even')
        cr_odd = ClassicalRegister(self.bit_precision, 'cr_odd')
        
        # 创建量子电路
        qc = QuantumCircuit(qr_even, qr_odd, cr_even, cr_odd)
        
        # 将像素值编码为量子态
        even_bits = [int(b) for b in bin(even_val)[2:].zfill(self.bit_precision)]
        odd_bits = [int(b) for b in bin(odd_val)[2:].zfill(self.bit_precision)]
        
        # 初始化量子比特
        for i in range(self.bit_precision):
            if even_bits[i]:
                qc.x(qr_even[i])
            if odd_bits[i]:
                qc.x(qr_odd[i])
        
        # 应用量子S变换门
        qc.barrier()
        qc.h(qr_even[0])  # Hadamard门创建叠加态
        qc.h(qr_odd[0])
        
        # CNOT门实现量子纠缠
        for i in range(self.bit_precision - 1):
            qc.cx(qr_even[i], qr_even[i+1])
            qc.cx(qr_odd[i], qr_odd[i+1])
            qc.cx(qr_even[i], qr_odd[i])
        
        # 量子S变换的核心操作
        for i in range(self.bit_precision - 4):
            # 创建自定义的S变换门
            qc.h(qr_even[i])
            qc.h(qr_odd[i])
            qc.cx(qr_even[i], qr_odd[i])
            qc.rz(np.pi/4, qr_even[i])
            qc.rz(np.pi/4, qr_odd[i])
        
        qc.barrier()
        
        # 测量量子比特
        qc.measure(qr_even, cr_even)
        qc.measure(qr_odd, cr_odd)
        
        return qc
    
    def execute_s_transform(self, even_val: int, odd_val: int):
        """执行量子S变换"""
        qc = self.create_s_transform_circuit(even_val, odd_val)
        
        # 使用Aer模拟器
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(qc, backend)
        job = backend.run(transpiled_qc, shots=1000)
        result = job.result()
        counts = result.get_counts()
        
        # 解析结果
        most_likely_result = max(counts, key=counts.get)
        
        # 分离even和odd的结果
        even_result = most_likely_result[:self.bit_precision]
        odd_result = most_likely_result[self.bit_precision:2*self.bit_precision]
        
        # 转换为十进制
        new_even_val = int(even_result, 2)
        new_odd_val = int(odd_result, 2)
        
        return new_even_val, new_odd_val, qc, counts


class QuantumKirschCircuit:
    """量子Kirsch边缘检测的量子电路实现"""
    
    def __init__(self):
        # 8个Kirsch方向模板
        self.kirsch_kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # North
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Northwest
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # West
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Southwest
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # South
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # Southeast
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # East
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])   # Northeast
        ]
    
    def create_kirsch_direction_circuit(self, pixel_values: np.ndarray, kernel: np.ndarray):
        """为单个Kirsch方向创建量子电路"""
        # 创建量子寄存器
        qr_pixels = QuantumRegister(9, 'pixels')  # 3x3像素块
        qr_kernel = QuantumRegister(9, 'kernel')  # 3x3核
        qr_result = QuantumRegister(8, 'result')  # 结果寄存器
        cr_result = ClassicalRegister(8, 'cr_result')
        
        qc = QuantumCircuit(qr_pixels, qr_kernel, qr_result, cr_result)
        
        # 初始化像素值
        for i, pixel_val in enumerate(pixel_values.flatten()):
            if pixel_val > 0:
                qc.x(qr_pixels[i])
        
        # 初始化核值
        for i, kernel_val in enumerate(kernel.flatten()):
            if kernel_val > 0:
                qc.x(qr_kernel[i])
        
        qc.barrier()
        
        # 量子卷积操作
        for i in range(9):
            qc.h(qr_pixels[i])
            qc.cx(qr_pixels[i], qr_kernel[i])
            qc.h(qr_kernel[i])
        
        # 量子测量和结果计算
        qc.barrier()
        qc.measure(qr_result, cr_result)
        
        return qc
    
    def quantum_kirsch_edge_detection(self, image: np.ndarray):
        """量子Kirsch边缘检测"""
        height, width = image.shape
        edge_result = np.zeros((height, width), dtype=np.uint8)
        
        # 为每个像素创建量子电路
        for i in range(1, height-1):
            for j in range(1, width-1):
                # 提取3x3邻域
                neighborhood = image[i-1:i+2, j-1:j+2]
                
                # 计算8个方向的响应
                responses = []
                for kernel in self.kirsch_kernels:
                    # 使用量子电路计算响应
                    qc = self.create_kirsch_direction_circuit(neighborhood, kernel)
                    
                    # 执行量子电路
                    backend = Aer.get_backend('qasm_simulator')
                    transpiled_qc = transpile(qc, backend)
                    job = backend.run(transpiled_qc, shots=100)
                    result = job.result()
                    counts = result.get_counts()
                    
                    # 解析结果
                    most_likely = max(counts, key=counts.get)
                    response = int(most_likely, 2)
                    responses.append(response)
                
                # 选择最大响应
                edge_result[i, j] = np.max(responses)
        
        # 归一化到0-255
        if edge_result.max() > 0:
            edge_result = (edge_result / edge_result.max()) * 255
        
        return edge_result.astype(np.uint8)


class QuantumCircuitVisualizer:
    """量子电路可视化器"""
    
    def __init__(self):
        self.backend = Aer.get_backend('statevector_simulator')
    
    def visualize_s_transform_circuit(self, even_val: int, odd_val: int):
        """可视化量子S变换电路"""
        s_transform = QuantumSTransformCircuit()
        new_even, new_odd, qc, counts = s_transform.execute_s_transform(even_val, odd_val)
        
        print(f"量子S变换结果:")
        print(f"原始值: even={even_val}, odd={odd_val}")
        print(f"变换后: even={new_even}, odd={new_odd}")
        
        # 绘制电路
        print("\n量子S变换电路:")
        print(qc)
        
        # 绘制测量结果直方图
        plt.figure(figsize=(12, 6))
        plot_histogram(counts)
        plt.title("量子S变换测量结果")
        plt.show()
        
        return qc
    
    def visualize_kirsch_circuit(self, pixel_values: np.ndarray, kernel: np.ndarray):
        """可视化量子Kirsch电路"""
        kirsch = QuantumKirschCircuit()
        qc = kirsch.create_kirsch_direction_circuit(pixel_values, kernel)
        
        print(f"量子Kirsch电路 (核: {kernel.flatten()})")
        print(qc)
        
        return qc


class QuantumImageProcessorCircuit:
    """基于量子电路的图像处理器"""
    
    def __init__(self, bit_precision: int = 8):
        self.s_transform = QuantumSTransformCircuit(bit_precision)
        self.kirsch = QuantumKirschCircuit()
        self.visualizer = QuantumCircuitVisualizer()
    
    def process_image_with_circuits(self, image: np.ndarray):
        """使用量子电路处理图像"""
        print("开始量子电路图像处理...")
        
        height, width = image.shape
        s_result = np.zeros_like(image)
        edge_result = np.zeros_like(image)
        
        # 处理2x2块进行S变换
        for i in range(0, height-1, 2):
            for j in range(0, width-1, 2):
                block = image[i:i+2, j:j+2]
                
                # 对每个像素对应用量子S变换
                for pair_idx in range(2):
                    even_val = int(block[pair_idx//2, pair_idx%2])
                    odd_val = int(block[pair_idx//2, (pair_idx+1)%2])
                    
                    # 执行量子S变换
                    new_even, new_odd, qc, counts = self.s_transform.execute_s_transform(even_val, odd_val)
                    
                    # 存储结果
                    s_result[i+pair_idx//2, j+pair_idx%2] = new_even
                    s_result[i+pair_idx//2, j+(pair_idx+1)%2] = new_odd
        
        # 量子Kirsch边缘检测
        edge_result = self.kirsch.quantum_kirsch_edge_detection(s_result)
        
        return s_result, edge_result


def load_and_preprocess_image(image_path: str):
    """加载和预处理图像"""
    image = Image.open(image_path)
    if image.mode != 'L':
        image = image.convert('L')
    image_array = np.array(image)
    print(f"图像尺寸: {image_array.shape}")
    print(f"像素值范围: {image_array.min()} - {image_array.max()}")
    return image_array


def visualize_circuit_results(original: np.ndarray, s_result: np.ndarray, edge_result: np.ndarray):
    """可视化量子电路处理结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("原始图像")
    axes[0].axis('off')
    
    axes[1].imshow(s_result, cmap='gray')
    axes[1].set_title("量子S变换结果")
    axes[1].axis('off')
    
    axes[2].imshow(edge_result, cmap='gray')
    axes[2].set_title("量子Kirsch边缘检测")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def demonstrate_quantum_circuits():
    """演示量子电路操作"""
    print("=" * 60)
    print("量子电路演示")
    print("=" * 60)
    
    # 演示量子S变换电路
    print("\n1. 量子S变换电路演示")
    visualizer = QuantumCircuitVisualizer()
    qc_s = visualizer.visualize_s_transform_circuit(100, 150)
    
    # 演示量子Kirsch电路
    print("\n2. 量子Kirsch电路演示")
    pixel_values = np.array([[100, 120, 110],
                            [130, 140, 125],
                            [115, 135, 145]])
    kernel = np.array([[5, 5, 5],
                      [-3, 0, -3],
                      [-3, -3, -3]])
    qc_k = visualizer.visualize_kirsch_circuit(pixel_values, kernel)


def main():
    """主函数：演示量子电路图像处理"""
    image_path = "1.1.01.tiff"
    
    print("=" * 60)
    print("量子电路图像处理系统")
    print("=" * 60)
    
    # 1. 加载图像
    print("\n1. 加载图像...")
    original_image = load_and_preprocess_image(image_path)
    
    # 2. 创建量子电路处理器
    print("\n2. 初始化量子电路处理器...")
    processor = QuantumImageProcessorCircuit(bit_precision=8)
    
    # 3. 演示量子电路
    print("\n3. 演示量子电路...")
    demonstrate_quantum_circuits()
    
    # 4. 处理图像
    print("\n4. 使用量子电路处理图像...")
    start_time = time.time()
    s_result, edge_result = processor.process_image_with_circuits(original_image)
    processing_time = time.time() - start_time
    
    # 5. 可视化结果
    print("\n5. 可视化量子电路处理结果...")
    visualize_circuit_results(original_image, s_result, edge_result)
    
    # 6. 保存结果
    print("\n6. 保存量子电路处理结果...")
    Image.fromarray(s_result).save("quantum_circuit_s_transform_result.tiff")
    Image.fromarray(edge_result).save("quantum_circuit_kirsch_edge_result.tiff")
    
    # 7. 输出统计信息
    print("\n" + "=" * 60)
    print("量子电路处理统计:")
    print(f"原始图像尺寸: {original_image.shape}")
    print(f"量子电路处理时间: {processing_time:.2f} 秒")
    print("结果文件:")
    print("- quantum_circuit_s_transform_result.tiff (量子S变换结果)")
    print("- quantum_circuit_kirsch_edge_result.tiff (量子Kirsch边缘检测结果)")
    print("=" * 60)
    print("量子电路图像处理完成!")


if __name__ == "__main__":
    main() 