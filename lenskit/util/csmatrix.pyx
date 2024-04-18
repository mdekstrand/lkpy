# cython: language_level=3str

cdef class CSMatrix:
    cdef readonly int nrows, ncols, nnz
    cdef readonly int[::1] rowptr
    cdef readonly int[::1] colind
    cdef readonly double[::1] values

    def __cinit__(self, int nr, int nc, int[::1] rps, int[::1] cis, double[::1] vs):
        self.nrows = nr
        self.ncols = nc
        self.rowptr = rps
        self.colind = cis
        self.values = vs
        self.nnz = self.rowptr[nr]

    @staticmethod
    def from_scipy(m):
        nr, nc = m.shape

        return CSMatrix(nr, nc, m.indptr, m.indices, m.data)

    cpdef (int,int) row_extent(self, row):
        if row < 0 or row >= self.nrows:
            raise IndexError(f"invalid row {row} for {self.nrows}x{self.ncols} matrix")

        return self.rowptr[row], self.rowptr[row+1]
