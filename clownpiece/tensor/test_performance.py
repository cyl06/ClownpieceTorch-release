import time
import tensor_impl as Tensor
import threading
from concurrent.futures import ThreadPoolExecutor

# --- 测试配置 ---
SIZES_TO_TEST_OP = [i for i in range(10, 1000, 20)] 
SIZES_TO_TEST_MAT = [i for i in range(100, 1000, 100)] 
WARMUP_RUNS = 1
TIMING_RUNS = 5

# # --- 调试配置 ---
# SIZES_TO_TEST_OP = [i for i in range(100, 1000, 100)] 
# SIZES_TO_TEST_MAT = [i for i in range(100, 1000, 100)] 
# WARMUP_RUNS = 1
# TIMING_RUNS = 1

def time_operation(op_name, op_lambda, size_info):
    
    """一个通用的计时函数"""
    # 1. 预热
    for _ in range(WARMUP_RUNS):
        op_lambda()
    
    # 2. 计时
    start_time = time.perf_counter()
    for _ in range(TIMING_RUNS):
        op_lambda()
    end_time = time.perf_counter()
    
    # 3. 计算并打印结果
    avg_duration_ms = ((end_time - start_time) / TIMING_RUNS) * 1000
    print(f"  - {op_name:<25} @ {size_info:<15}: {avg_duration_ms:.4f} ms")

def main():
    print("="*60)
    print(f"开始性能测试 ({Tensor.parallel_strategy()})")
    print("="*60)

    for size in SIZES_TO_TEST_OP:
        print(f"\n--- 测试尺寸: {size}x{size} ---")
        x = Tensor.randn([size, size])
        elementwise_op = lambda: x * 2.0 + 3.0
        time_operation("Element-wise (y=x*2+3)", elementwise_op, f"{size}x{size}")
    
    for size in SIZES_TO_TEST_MAT:
        print(f"\n--- 测试尺寸: {size}x{size} ---")
        a = Tensor.randn([size, size])
        b = Tensor.randn([size, size])
        matmul_op = lambda: a @ b
        time_operation("Matrix Multiplication", matmul_op, f"{size}x{size}")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()