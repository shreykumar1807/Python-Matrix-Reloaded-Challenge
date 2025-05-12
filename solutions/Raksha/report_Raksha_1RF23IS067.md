# Matrix Operations Challenge

This project implements a `Matrix` class in Python supporting various operations including addition, subtraction, element-wise multiplication, matrix multiplication, and exponentiation - all with broadcasting support.

## Features

- Basic arithmetic operations (`+`, `-`, `*`, `@`, `**`)
- Broadcasting support (similar to NumPy)
- Memory optimization using `__slots__` and weak references
- Cython acceleration for critical operations
- Asynchronous processing capabilities
- Performance profiling and analysis tools

## Class Structure

The `Matrix` class provides a wrapper around NumPy arrays with the following operations:

| Method       | Operation             | Description                        |
|--------------|------------------------|------------------------------------|
| `__add__`    | `A + B`                | Addition with broadcasting          |
| `__sub__`    | `A - B`                | Subtraction with broadcasting       |
| `__mul__`    | `A * B`                | Element-wise multiplication         |
| `__matmul__` | `A @ B`                | Matrix multiplication               |
| `__pow__`    | `A ** n`               | Element-wise exponentiation         |

## Usage Example

```python
# Initialize matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5], [6]])  # Will broadcast to match A's shape

# Perform complex operations
result = (A + B) @ (A - B) ** 2
print(result)
```

## Building the Cython Extension

The project includes Cython-optimized implementations of critical matrix operations. To build the extension:

```bash
python setup.py build_ext --inplace
```

## Performance Benchmarking

To compare the performance of the pure Python implementation with the Cython-accelerated version:

```bash
python benchmark.py
```

This will run a series of benchmarks with different matrix sizes and report the relative speedup achieved with Cython.

## Performance Analysis

The script includes several performance analysis tools:

1. **Timing Measurement**: Measures the execution time of operations
2. **Memory Usage Tracking**: Uses `tracemalloc` to monitor memory consumption
3. **Function Profiling**: Uses `cProfile` to identify bottlenecks
4. **Line Profiling**: Supports line-by-line analysis with `line_profiler`

## Running the Code

```bash
# Basic execution
python matrix_challenge.py

# With line profiling (requires line_profiler package)
pip install line_profiler
kernprof -l -v matrix_challenge.py

# Benchmark Cython vs Pure Python
python benchmark.py
```

## Optimizations Applied

1. **Memory Efficiency**:
   - Used `__slots__` to reduce memory overhead
   - Weak references for caching
   - Avoided unnecessary data copies

2. **Performance**:
   - Cython implementation for critical operations
   - Asynchronous processing
   - Leveraged NumPy's optimized functions
   - Optimized broadcasting implementations

3. **Code Quality**:
   - List comprehensions instead of loops
   - Closure-based optimization
   - LRU caching for repeated operations

## Requirements

- Python 3.6+
- NumPy
- Cython
- Optional: `line_profiler` (for detailed profiling)
Output:

=== Matrix Operations Challenge ===
Using Cython acceleration: False
=== Evaluating expression: (A + B) @ (A - B) ** 2 ===
Matrix A:
[[1 2]
 [3 4]]
Matrix B:
[[5]
 [6]]

Result:
[[159  82]
 [234 121]]

Reference NumPy result:
[[159  82]
 [234 121]]
Results match: True

=== Performance Measurements ===
Average execution time (100 runs): 0.000048 seconds

=== Memory Usage Analysis ===
Memory before: 0.00 KB, peak: 0.00 KB
Memory after: 0.16 KB, peak: 1.26 KB
Difference: 0.16 KB

=== cProfile Analysis ===
         26024 function calls in 0.074 seconds

   Ordered by: cumulative time
   List reduced from 24 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     4000    0.015    0.000    0.033    0.000 C:\Users\ADMIN\Desktop\matrix_challenge.py:79(__init__)
     1000    0.007    0.000    0.025    0.000 C:\Users\ADMIN\Desktop\matrix_challenge.py:161(__pow__)
    18000    0.020    0.000    0.020    0.000 {built-in method builtins.isinstance}
     1000    0.013    0.000    0.018    0.000 C:\Users\ADMIN\Desktop\matrix_challenge.py:100(__add__)
     1000    0.009    0.000    0.017    0.000 C:\Users\ADMIN\Desktop\matrix_challenge.py:145(__matmul__)
     1000    0.008    0.000    0.013    0.000 C:\Users\ADMIN\Desktop\matrix_challenge.py:116(__sub__)
        1    0.000    0.000    0.009    0.009 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\queue.py:180(get)
        1    0.001    0.001    0.009    0.009 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\threading.py:327(wait)
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
        2    0.000    0.000    0.000    0.000 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\queue.py:267(_qsize)
        1    0.000    0.000    0.000    0.000 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\threading.py:315(_acquire_restore)
        1    0.000    0.000    0.000    0.000 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\threading.py:306(__exit__)
        3    0.000    0.000    0.000    0.000 {method 'acquire' of '_thread.lock' objects}
        1    0.000    0.000    0.000    0.000 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\threading.py:303(__enter__)
        1    0.000    0.000    0.000    0.000 c:\Users\ADMIN\AppData\Local\Programs\Python\Python313\Lib\threading.py:312(_release_save)


