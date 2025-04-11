use numpy::prelude::*;
use pyo3::prelude::*;

use numpy::PyArray1;

/// Python handle to a compressed sparse row matrix.
#[pyclass]
pub struct CSRMatrix {
    pub(crate) n_rows: u32,
    pub(crate) n_cols: u32,
    pub(crate) rowptrs: Py<PyArray1<i64>>,
    pub(crate) colinds: Py<PyArray1<i32>>,
    pub(crate) values: Py<PyArray1<f32>>,
}

#[pyfunction]
pub fn make_csr<'py>(
    py: Python<'py>,
    rps: &Bound<'py, PyArray1<i64>>,
    cis: &Bound<'py, PyArray1<i32>>,
    vs: &Bound<'py, PyArray1<f32>>,
    shape: (u32, u32),
) -> PyResult<Bound<'py, CSRMatrix>> {
    let (n_rows, n_cols) = shape;
    Bound::new(
        py,
        CSRMatrix {
            n_rows,
            n_cols,
            rowptrs: rps.clone().unbind(),
            colinds: cis.clone().unbind(),
            values: vs.clone().unbind(),
        },
    )
}

#[pymethods]
impl CSRMatrix {
    #[getter]
    fn nnz<'py>(&self, py: Python<'py>) -> u64 {
        self.colinds.bind(py).len() as u64
    }
    #[getter]
    fn nrow<'py>(&self) -> u32 {
        self.n_rows
    }
    #[getter]
    fn ncol<'py>(&self) -> u32 {
        self.n_cols
    }

    #[getter]
    fn rowptrs<'py>(&self, py: Python<'py>) -> &Bound<'py, PyArray1<i64>> {
        self.rowptrs.bind(py)
    }

    #[getter]
    fn colinds<'py>(&self, py: Python<'py>) -> &Bound<'py, PyArray1<i32>> {
        self.colinds.bind(py)
    }

    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> &Bound<'py, PyArray1<f32>> {
        self.values.bind(py)
    }
}
