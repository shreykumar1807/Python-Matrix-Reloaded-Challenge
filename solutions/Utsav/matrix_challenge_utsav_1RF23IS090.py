#!pip install line_profiler

import numpy as np
import time
import tracemalloc

class Matrix:
    __slots__ = ['data']
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data, np.ndarray):
            self.data = data.astype(np.float64, copy=False)

    def __add__(self, other):
        return Matrix(self.data + (other.data if isinstance(other, Matrix) else other))

    def __sub__(self, other):
        return Matrix(self.data - (other.data if isinstance(other, Matrix) else other))

    def __mul__(self, other):
        return Matrix(self.data * (other.data if isinstance(other, Matrix) else other))

    def __matmul__(self, other):
        return Matrix(self.data @ other.data)

    def __pow__(self, power):
        return Matrix(np.power(self.data, power))

    def __str__(self):
        return str(self.data)

A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("Add:", A + 3)
print("Subtract:", A - 2)
print("Multiply:", A * B)
print("Power:", A ** 2)
print("Matrix Multiplication:", A @ A)

tracemalloc.start()
snapshot1 = tracemalloc.take_snapshot()

start_time = time.perf_counter()
result = (A + B) @ (A - B) ** 2
end_time = time.perf_counter()

snapshot2 = tracemalloc.take_snapshot()

print("Complex Expression Result:", result)
print("Calculation took", end_time - start_time, "seconds")

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
for stat in top_stats[:3]:
    print(stat)
