## ✅ Summary of Tasks

| Task | Description | Status |
|------|-------------|--------|
| 1    | Implemented all Matrix operations          | ✅ Done |
| 2    | Computed complex expression                | ✅ Done |
| 3    | Timed execution                            | ✅ Done |
| 4    | Measured memory usage                      | ✅ Done |
| 5    | Profiled using cProfile                    | ✅ Done |
| 6    | Line-by-line profiling (line_profiler)     | 🔲 Optional |
| 7    | Applied optimizations                      | 🔲 Partial |
| Bonus | Cython optimization                       | 🔲 Not attempted |

---

## 🧪 Task 1: Matrix Operations Implementation

Implemented the following methods with broadcasting support:

- `__add__`
- `__sub__`
- `__mul__`
- `__matmul__`
- `__pow__`

Used NumPy internally for performance and `__slots__` to reduce memory overhead.

---

## 🧠 Task 2: Complex Expression

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([5, 6])  # Broadcasts to (2, 2)
result = (A + B) @ (A - B) ** 2
```

### ✅ Output:
```
[Paste printed output here]
```

---

## ⏱️ Task 3: Execution Time

```text
Execution Time: [insert measured time here] seconds
```

- Measured using `time.perf_counter()` for better precision.
- Expression was run once in the script.

---

## 📦 Task 4: Memory Usage

Measured with `tracemalloc`:

```
Current memory usage: [xx.xx] KB
Peak memory usage:    [xx.xx] KB
```

---

## 📊 Task 5: Profiling Output (`cProfile`)

```text
[Paste or screenshot cProfile output here]

Example:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       2    0.000    0.000    0.001    0.000 matrix_challenge.py:50(__add__)
       1    0.000    0.000    0.001    0.001 matrix_challenge.py:81(run_expression)
...
```

**Analysis:**
- Most time spent in NumPy operations inside `__add__`, `__sub__`, and `__matmul__`.
- No major performance bottlenecks found.

---

## 🔍 Task 6: Line-by-Line Profiling (Optional)

> Requires running:
> ```bash
> kernprof -l -v matrix_challenge.py
> ```

### Example Output (if available):
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

## 🚀 Task 7: Optimizations

### ✅ Applied:
- Used `__slots__` to reduce object memory overhead.
- Avoided unnecessary object copies in overloaded operators.

### 🔲 Possible Improvements (Future Work):
- Implement Cython version of matrix arithmetic
- Replace repetitive operator code with internal helper functions
- Use `np.empty` or `np.zeros` where applicable to preallocate

---

## 💡 Bonus: Cython Rewrite

> Not attempted in this version. Could be explored in future for performance gains in tight loops or element-wise operations.

---

## 📎 Appendix

- Filename: `matrix_challenge.py`
- Python version: `Python 3.x`
- NumPy version: `[insert here]`
- All required packages are standard or installable via pip
