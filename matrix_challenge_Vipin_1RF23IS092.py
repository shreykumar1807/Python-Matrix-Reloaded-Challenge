import numpy as np
import time
import tracemalloc
import cProfile

class Matrix:
    __slots__ = ['data']

    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a list or NumPy array")
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        if data.ndim != 2:
            raise ValueError("Only 2D matrices are allowed")
        self.data = data

    def __add__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data + other.data)
        return Matrix(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data - other.data)
        return Matrix(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data * other.data)
        return Matrix(self.data * other)

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("Matrix multiplication requires another Matrix")
        return Matrix(self.data @ other.data)

    def __pow__(self, power):
        return Matrix(self.data ** power)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return self.data.shape


# this is Complex expression: (A + B) @ (A - B) ** 2
A = Matrix([[1, 2], [3, 4]])
B = Matrix([5, 6]) 

# this is Timing execution
start = time.perf_counter()
result = (A + B) @ (A - B) ** 2
end = time.perf_counter()

print("Result:\n", result)
print(f"\nExecution Time: {end - start:.6f} seconds")

# this is for Memory usage
tracemalloc.start()
_ = (A + B) @ (A - B) ** 2
current, peak = tracemalloc.get_traced_memory()
print(f"\nCurrent memory usage: {current / 1024:.2f} KB")
print(f"Peak memory usage: {peak / 1024:.2f} KB")
tracemalloc.stop()

# this is Profiling with cProfile
print("\n--- cProfile Output ---")
cProfile.run("(A + B) @ (A - B) ** 2")
