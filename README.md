
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



## 📅 Deadline

**🕘 Submit before: 9 PM today**



## 📥 Submission Instructions

1. **Fork** this repository.
2. Add your completed files to your forked repository.
3. **Name your files using the following format**:

```
matrix_challenge_<Name>_<USN>.py
report_<Name>_<USN>.md
```

4. **Create a Pull Request** to this repository **before the deadline**.



## 🗂️ Files to Submit

- `matrix_challenge_<Name>_<USN>.py`
- `report_<Name>_<USN>.md`

Your report should include:

- Profiling outputs (`cProfile`, `line_profiler`)
- Time and memory comparisons (before vs after optimization)
- Explanation of optimizations and their impact



## 💎 Bonus

Re-implement part of your Matrix class in **Cython** for additional performance gains.



Happy Hacking! 🧠💻
