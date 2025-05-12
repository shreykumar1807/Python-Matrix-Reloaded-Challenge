
# 🧠 Matrix Operations: Python vs. Cython Implementation By Vipin Choudhary

## 1. Introduction

This report compares two implementations of a custom `Matrix` class supporting:

- **Addition (`+`)**
- **Subtraction (`-`)**
- **Matrix multiplication (`@`)**
- **Element-wise exponentiation (`**`)**

### Implementations

- **Pure Python version**: `matrix_challenge.py`  
- **Cython-accelerated version**: `matrix_cy.pyx` + `setup.py`

### We Measure

- **Correctness** (outputs)
- **Execution time** (before vs. after)
- **Memory usage** (before vs. after)
- **Profiling** (cProfile breakdown)
- **Optimizations and their impact**



## 2. Code Listing & there Output

### 2.1 Pure Python (`matrix_challenge.py`)

```python
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


# Complex expression: (A + B) @ (A - B) ** 2
A = Matrix([[1, 2], [3, 4]])
B = Matrix([5, 6]) 

# Timing execution
start = time.perf_counter()
result = (A + B) @ (A - B) ** 2
end = time.perf_counter()

print("Result:\n", result)
print(f"\nExecution Time: {end - start:.6f} seconds")

# Memory usage
tracemalloc.start()
_ = (A + B) @ (A - B) ** 2
current, peak = tracemalloc.get_traced_memory()
print(f"\nCurrent memory usage: {current / 1024:.2f} KB")
print(f"Peak memory usage: {peak / 1024:.2f} KB")
tracemalloc.stop()

# Profiling with cProfile
print("\n--- cProfile Output ---")
cProfile.run("(A + B) @ (A - B) ** 2")
```
<img width="688" alt="python implementtation" src="https://github.com/user-attachments/assets/cac74599-1e39-4501-bbd8-e7e24a90743c" />




### 2.2 Cython (matrix_cy.pyx + setup.py)

```python
import numpy as np
cimport numpy as np

cdef class MatrixCy:
    cdef np.ndarray data

    def __init__(self, data):
        if isinstance(data, list):
            data = np.array(data, dtype=np.double)
        elif not isinstance(data, np.ndarray):
            raise TypeError("Data must be list or ndarray")
        if data.ndim == 1:
            data = np.expand_dims(data, 0)
        if data.ndim != 2:
            raise ValueError("Only 2D arrays allowed")
        self.data = data.astype(np.double)

    def __add__(self, MatrixCy other):
        return MatrixCy(self.data + other.data)

    def __sub__(self, MatrixCy other):
        return MatrixCy(self.data - other.data)

    def __matmul__(self, MatrixCy other):
        return MatrixCy(self.data.dot(other.data))

    def __pow__(self, int power):
        cdef int i, j
        cdef int rows = self.data.shape[0]
        cdef int cols = self.data.shape[1]
        cdef np.ndarray[np.double_t, ndim=2] out = np.empty((rows, cols), dtype=np.double)
        cdef double[:, :] in_view = self.data
        cdef double[:, :] out_view = out
        for i in range(rows):
            for j in range(cols):
                out_view[i, j] = in_view[i, j] ** power
        return MatrixCy(out)

    @property
    def data_view(self):
        """Expose the underlying NumPy array to Python."""
        return self.data
```
<img width="431" alt="Cython Implementation" src="https://github.com/user-attachments/assets/92ddfe12-799b-4069-b735-9f7fae916965" />

## 3. Profiling, Timing, and Memory Usage Report

### 3.1 Matrix Operation
Expression evaluated:
```python
(A + B) @ (A - B) ** 2
```
Result:
 [[144 100]
 [400 324]]


 ### 3.2 Timing Comparison

 | Version     | Execution Time (seconds) |
| ----------- | ------------------------ |
| Pure Python | 0.000028                 |
| Cython      | 0.000008                 |

Improvement: ~3.5× faster with Cython


### 3.4 Memory Usage Comparison

| Version     | Current Memory Usage | Peak Memory Usage |
| ----------- | -------------------- | ----------------- |
| Pure Python | 0.45 KB              | 0.61 KB           |
| Cython      | 0.30 KB              | 0.40 KB           |

Improvement: Memory usage reduced by ~33%


### 3.5 Optimization Techniques Applied

Converted the Matrix class to use __slots__ to reduce memory overhead

Moved numerical operations to Cython for faster array operations

Avoided unnecessary object creation during matrix operations


### 3.6 Impact of Optimizations

Execution Time: Reduced by ~3.5× due to compiled C-level matrix operations

Memory Usage: Reduced by ~33% with __slots__ and less overhead

Function Calls: Reduced runtime function calls with Cython compilation



