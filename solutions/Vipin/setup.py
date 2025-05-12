from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="matrix_cy",
    ext_modules=cythonize("matrix_cy.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
