import numpy as np

class Matrix:
    _slots_ = ['data']

    def _init_(self, data):
        if isinstance(data, list):
            data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a list or NumPy array")
        if data.ndim != 2:
            data = data.reshape(1, -1)  # Convert 1D to 2D row
        self.data = data

    def _add_(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data + other.data)
        return Matrix(self.data + other)

    def _sub_(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data - other.data)
        return Matrix(self.data - other)

    def _mul_(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data * other.data)
        return Matrix(self.data * other)

    def _matmul_(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        return Matrix(self.data @ other)

    def _pow_(self, power):
        return Matrix(self.data ** power)

    def _str_(self):
        return str(self.data)

    def _repr_(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return self.data.shape
import time
import tracemalloc
import cProfile

A = Matrix([[1, 2], [3, 4]])
B = Matrix([5, 6])

# --- Timing ---
start = time.perf_counter()
result = (A + B) @ ((A - B) ** 2)
end = time.perf_counter()

print("Result:\n", result)
print(f"Execution Time: {end - start:.6f} seconds")

# --- Memory ---
tracemalloc.start()
_ = (A + B) @ ((A - B) ** 2)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024:.2f} KB; Peak: {peak / 1024:.2f} KB")
tracemalloc.stop()

# --- Profiling ---
print("\nProfiling with cProfile:")
cProfile.run("(A + B) @ ((A - B) ** 2)")
