import numpy as np
import time
import tracemalloc
import cProfile
from line_profiler import LineProfiler

class Matrix:
    __slots__ = ['data']

    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            data = data.astype(np.float64)
        else:
            raise TypeError("Data must be a list or NumPy array")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError("Only 1D or 2D matrices are allowed")
            
        self.data = data

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data + other.data)
        elif isinstance(other, (int, float)):
            return Matrix(self.data + other)
        else:
            return Matrix(self.data + np.array(other))

    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data - other.data)
        elif isinstance(other, (int, float)):
            return Matrix(self.data - other)
        else:
            return Matrix(self.data - np.array(other))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data * other.data)
        elif isinstance(other, (int, float)):
            return Matrix(self.data * other)
        else:
            return Matrix(self.data * np.array(other))

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        else:
            return Matrix(self.data @ np.array(other))

    def __pow__(self, power):
        return Matrix(self.data ** power)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return self.data.shape

# Global test matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([5, 6])

def demonstrate_complex_expression():
    result = (A + B) @ (A - B) ** 2
    print("Result of (A + B) @ (A - B) ** 2:")
    print(result)
    
    np_A = np.array([[1, 2], [3, 4]], dtype=np.float64)
    np_B = np.array([5, 6], dtype=np.float64)
    np_result = (np_A + np_B) @ (np_A - np_B) ** 2
    print("\nVerification with numpy:")
    print(np_result)
    
    return result

def time_execution():
    # Warm-up
    _ = (A + B) @ (A - B) ** 2
    
    start = time.perf_counter()
    for _ in range(1000):
        result = (A + B) @ (A - B) ** 2
    end = time.perf_counter()
    
    avg_time = (end - start) / 1000
    print(f"\nAverage execution time over 1000 runs: {avg_time:.6f} seconds")
    return result, avg_time

def measure_memory():
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    result = (A + B) @ (A - B) ** 2
    
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    print("\nMemory allocation difference:")
    for stat in top_stats[:5]:
        print(stat)
    
    tracemalloc.stop()
    return result

def profile_functions():
    print("\nProfiling with cProfile:")
    cProfile.runctx(
        'result = (A + B) @ (A - B) ** 2',
        globals(),
        {'A': A, 'B': B},
        sort='cumtime'
    )

def line_by_line_profile():
    lp = LineProfiler()
    lp.add_function(demonstrate_complex_expression)
    
    print("\nLine-by-line profiling:")
    lp_wrapper = lp(demonstrate_complex_expression)
    lp_wrapper()
    lp.print_stats()

   

if __name__ == "__main__":
    # Task 2
    result = demonstrate_complex_expression()
    
    # Task 3
    _, exec_time = time_execution()
    
    # Task 4
    _ = measure_memory()
    
    # Task 5
    profile_functions()
    
    # Task 6
    line_by_line_profile()