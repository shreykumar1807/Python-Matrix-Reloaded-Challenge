
import numpy as np

class Matrix:
    """
    A 2D Matrix class supporting basic operations and broadcasting.
    Internally uses NumPy for performance but encapsulates it for educational purposes.
    """

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
        if isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        return Matrix(self.data @ other)

    def __pow__(self, power):
        return Matrix(self.data ** power)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix({repr(self.data)})"

    def shape(self):
        return self.data.shape


# Demonstration script (task 2–4)
if __name__ == "__main__":
    import time
    import tracemalloc
    import cProfile

    def run_expression():
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([5, 6])  # Will be broadcasted to shape (2, 2)

        start_time = time.perf_counter()
        tracemalloc.start()

        result = (A + B) @ (A - B) ** 2
        print("Result:", result)

        current, peak = tracemalloc.get_traced_memory()
        end_time = time.perf_counter()

        print(f"Execution Time: {end_time - start_time:.6f} seconds")
        print(f"Current memory usage: {current / 1024:.2f} KB")
        print(f"Peak memory usage: {peak / 1024:.2f} KB")
        tracemalloc.stop()

    cProfile.run("run_expression()")
