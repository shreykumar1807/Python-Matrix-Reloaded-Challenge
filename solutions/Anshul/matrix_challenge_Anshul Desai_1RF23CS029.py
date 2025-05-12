import numpy as np

class Matrix:
    """
    Custom 2D Matrix class supporting arithmetic operations with broadcasting.
    Designed for educational profiling and optimization tasks.
    """

    __slots__ = ['_data'] # Using __slots__ to reduce memory footprint by avoiding __dict__

    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            raise TypeError("Matrix data must be list or np.ndarray")

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim != 2:
            raise ValueError("Only 2D data is allowed")

        self._data = data

    @property
    def shape(self):
        return self._data.shape

    def add(self, other):
        return Matrix(self._data + self._get_data(other))

    def sub(self, other):
        return Matrix(self._data - self._get_data(other))

    def mul(self, other):
        return Matrix(self._data * self._get_data(other))

    def dot(self, other):
        return Matrix(self._data @ self._get_data(other))

    def power(self, n):
        return Matrix(self._data ** n)

    def _get_data(self, other):     # Internal utility to extract NumPy array from input (Matrix, list, scalar, or ndarray)
        if isinstance(other, Matrix):
            return other._data
        elif isinstance(other, (np.ndarray, list, int, float)):
            return np.array(other)
        raise TypeError("Unsupported type for Matrix operation")

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return f"Matrix({self._data.tolist()})"

# Test script and profiling
if __name__ == "__main__":
    import time
    import tracemalloc
    import cProfile

    def run_expression():
        A = Matrix([[1, 2], [3, 4]])
        B = Matrix([5, 6])

        tracemalloc.start()
        start = time.perf_counter()

        result = A.add(B).dot(A.sub(B).power(2))
        print("Result:\n", result)

        end = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"\nExecution Time: {end - start:.6f}s")
        print(f"Current Memory Usage: {current / 1024:.2f} KB")
        print(f"Peak Memory Usage: {peak / 1024:.2f} KB")

    # Basic profiling run to check if the efficiency of the various parts of the code 
    cProfile.run("run_expression()")
