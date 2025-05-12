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
