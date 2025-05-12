import numpy as np
import time
import tracemalloc
import cProfile
import pstats

# Import line_profiler's decorator
from line_profiler import profile

class Matrix:
    __slots__ = ['data']

    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a list or NumPy array")
        if data.ndim == 1:
            data = data.reshape(1, -1)  # Allow broadcasting
        elif data.ndim != 2:
            raise ValueError("Only 1D or 2D matrices are allowed")
        self.data = data

    @profile
    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data + other.data)
        else:
            return Matrix(self.data + other)

    @profile
    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data - other.data)
        else:
            return Matrix(self.data - other)

    @profile
    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data * other.data)
        else:
            return Matrix(self.data * other)

    @profile
    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Matrix multiplication requires another Matrix")
        return Matrix(self.data @ other.data)

    @profile
    def __pow__(self, power):
        return Matrix(np.power(self.data, power))

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return self.data.shape

# Test complex expression
@profile
def test_expression():
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([5, 6])  # Now allowed due to 1D → 2D conversion
    result = (A + B) @ (A - B) ** 2
    print("Result of (A + B) @ (A - B) ** 2:")
    print(result)
    return result

# Timing and memory measurement
@profile
def measure_performance():
    tracemalloc.start()
    start_time = time.perf_counter()

    for _ in range(1000):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([5, 6])
        result = (A + B) @ (A - B) ** 2

    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nTime per iteration: {(end_time - start_time)/1000:.6f} seconds")
    print(f"Memory usage: {peak / 1024:.2f} KB")

# cProfile for bottlenecks
@profile
def profile_cprofile():
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(1000):
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([5, 6])
        result = (A + B) @ (A - B) ** 2
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME).print_stats(5)

if __name__ == "__main__":
    test_expression()
    measure_performance()
    print("\ncProfile Results:")
    profile_cprofile()
