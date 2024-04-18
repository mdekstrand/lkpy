# cython: language_level=3str

cdef class CSMatrix:
    cdef readonly int nrows, ncols, nnz
    cdef readonly int[::1] rowptr
    cdef readonly int[::1] colind
    cdef readonly double[::1] values

    # void __cinit__(self, int nr, int nc, int[:] rps, int[:] cis, double[:] vs)

    cdef (int,int) row_extent(self, row)
