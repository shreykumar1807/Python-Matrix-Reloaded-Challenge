from matrix_cy import MatrixCy

a = MatrixCy([[1, 2], [3, 4]])
b = MatrixCy([[5, 6], [7, 8]])

print("a + b =\n", (a + b).data_view)
print("a - b =\n", (a - b).data_view)
print("a @ b =\n", (a @ b).data_view)
print("a ** 2 =\n", (a ** 2).data_view)
