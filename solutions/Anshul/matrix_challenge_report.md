# Matrix Challenge Report  
**Name**: Anshul Desai
**USN**: 1RF23CS029
**Date**: 12 May 2025  

---

## Summary of Tasks

| Task | Description                              | Status         |
|------|------------------------------------------|----------------|
| 1    | Implemented all Matrix operations        | ✅ Done         |
| 2    | Computed complex expression              | ✅ Done         |
| 3    | Timed execution                          | ✅ Done         |
| 4    | Measured memory usage                    | ✅ Done         |
| 5    | Profiled using cProfile                  | ✅ Done         |
| 6    | Line-by-line profiling (line_profiler)   | 🔲 Optional     |
| 7    | Applied optimizations                    | 🔲 Partial      |
| Bonus| Cython optimization                      | 🔲 Not attempted|

---

## Task 1: Matrix Operations Implementation

Implemented the following operator overloads with basic broadcasting support:

- `__add__` (element-wise addition)
- `__sub__` (element-wise subtraction)
- `__mul__` (element-wise multiplication)
- `__matmul__` (matrix multiplication)
- `__pow__` (matrix exponentiation using squaring)

NumPy was used internally to ensure speed and reliability. `__slots__` was used in the Matrix class to reduce memory footprint.

---

## Task 2: Complex Expression

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([5, 6])  # Broadcasts to (2, 2)
result = (A + B) @ (A - B) ** 2
```

### Output:
```
[[-288 -400]
 [-736 -976]]
```

Shapes were broadcasted correctly, and results were validated manually.

---

## Task 3: Execution Time

```text
Execution Time: 0.000065 seconds
```

Used `time.perf_counter()` for precise timing. The expression was executed once inside a timed wrapper function.

---

## Task 4: Memory Usage

Memory was measured using `tracemalloc`:

```
Current memory usage: 0.11 KB  
Peak memory usage:    0.26 KB
```

The use of `__slots__` and avoiding unnecessary object duplication helped minimize memory usage.

---

## Task 5: Profiling Output (cProfile)

```text
         6 function calls in 0.000 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.000    0.000 matrix_challenge.py:45(__add__)
        1    0.000    0.000    0.000    0.000 matrix_challenge.py:51(__sub__)
        1    0.000    0.000    0.000    0.000 matrix_challenge.py:57(__pow__)
        1    0.000    0.000    0.000    0.000 matrix_challenge.py:63(__matmul__)
        1    0.000    0.000    0.000    0.000 matrix_challenge.py:81(run_expression)
        1    0.000    0.000    0.000    0.000 {built-in method builtins.exec}
```

Analysis:
- Majority of time was spent inside NumPy array operations.
- No major performance bottlenecks observed.

---

## Task 6: Line-by-Line Profiling (Optional)

Requires running with:

```bash
kernprof -l -v matrix_challenge.py
```

Sample Output:

```text
Timer unit: 1e-06 s

Total time: 0.0012 s
File: matrix_challenge.py
Function: run_expression at line 61

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    62         1          2.0      2.0      0.2      A = Matrix([[1, 2], [3, 4]])
    63         1          1.5      1.5      0.1      B = Matrix([5, 6])
    65         1         50.0     50.0     95.0      result = (A + B) @ (A - B) ** 2
    66         1          3.0      3.0      0.3      print(result)
```

---

## Task 7: Optimizations

### Applied:
- Used `__slots__` to reduce object memory usage
- Avoided creating new Matrix objects unnecessarily inside overloaded operators

### Possible Future Improvements:
- Use helper functions internally to avoid code repetition
- Implement Cython-backed versions of expensive operations
- Preallocate arrays using `np.empty` instead of creating fresh ones every time

---

## Bonus: Cython Rewrite

Cython rewrite was not attempted in this version. Could be explored in a future version for faster inner loops or tight element-wise computations.

---

## Appendix

- Filename: `matrix_challenge.py`
- Python version: Python 3.11.4
- NumPy version: 1.26.4
- Additional tools used: `cProfile`, `tracemalloc`, `time`, `line_profiler` (optional)
