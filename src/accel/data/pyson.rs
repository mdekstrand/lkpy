// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for reading invalid JSON that is actually valid Python expression syntax.

use std::any::Any;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::Lines;
use std::thread;

use log::*;

use flate2::read::MultiGzDecoder;
use os_pipe::{pipe, PipeReader};
use pyo3::exceptions::PyIOError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::exceptions::PyStopIteration;
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyNone, PyString},
};

use rustpython_ast::{Constant, Expr, ExprDict, ExprList};
use rustpython_parser::Parse;

#[pyclass]
pub struct NDPysonReader {
    lines: Lines<BufReader<PipeReader>>,
    thread: Option<thread::JoinHandle<io::Result<u64>>>,
}

#[pymethods]
impl NDPysonReader {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'py>(mut slf: PyRefMut<'_, Self>, py: Python<'py>) -> PyResult<PyObject> {
        let line = slf.lines.next();
        if let Some(line) = line {
            let line = line?;
            pyson_loads(py, &line)
        } else {
            if let Some(thread) = slf.thread.take() {
                let res = thread.join();
                match res {
                    Ok(Ok(n)) => {
                        debug!("decompressed {} bytes", n);
                    }
                    Ok(Err(e)) => return Err(PyIOError::new_err(format!("IO error: {:?}", e))),
                    Err(e) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "failed to join backend thread: {:?}",
                            e
                        )))
                    }
                }
            }
            Err(PyStopIteration::new_err("decoding complete"))
        }
    }
}

#[pyfunction]
pub fn read_ndpyson(path: &str) -> PyResult<NDPysonReader> {
    let file = File::open(path)?;
    let (reader, writer) = pipe()?;

    let thread = thread::spawn(move || {
        let mut writer = writer;
        let mut read = MultiGzDecoder::new(file);
        io::copy(&mut read, &mut writer)
    });

    let reader = BufReader::new(reader);
    let lines = reader.lines();

    Ok(NDPysonReader {
        lines,
        thread: Some(thread),
    })
}

/// Parse a “pyson” object.
#[pyfunction]
pub fn pyson_loads<'py>(py: Python<'py>, text: &str) -> PyResult<PyObject> {
    let ast = ExprDict::parse(text, "internal")
        .map_err(|_e| PyErr::new::<PyValueError, _>("Python parse error"))?;
    let obj = realize_dict(py, ast)?;
    Ok(obj.unbind())
}

fn realize_value<'py>(py: Python<'py>, ast: Expr) -> PyResult<Bound<'py, PyAny>> {
    match ast {
        Expr::Constant(c) => realize_constant(py, c.value),
        Expr::List(list) => realize_list(py, list),
        Expr::Dict(dict) => realize_dict(py, dict),
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "unsupported expression type {:?}",
            ast.type_id()
        ))),
    }
}

fn realize_constant<'py>(py: Python<'py>, c: Constant) -> PyResult<Bound<'py, PyAny>> {
    match c {
        Constant::Bool(val) => Ok(PyBool::new(py, val).to_owned().into_any()),
        Constant::Bytes(val) => Ok(PyString::new(py, str::from_utf8(&val)?).into_any()),
        Constant::None => Ok(PyNone::get(py).to_owned().into_any()),
        Constant::Str(s) => Ok(PyString::new(py, &s).into_any()),
        Constant::Int(big_int) => {
            let n: i64 = big_int
                .try_into()
                .map_err(|_e| PyErr::new::<PyValueError, _>("integer out of bounds"))?;
            Ok(PyInt::new(py, n).into_any())
        }
        Constant::Tuple(constants) => {
            let list = PyList::empty(py);
            for elt in constants {
                list.append(realize_constant(py, elt)?)?;
            }
            Ok(list.into_any())
        }
        Constant::Float(x) => Ok(PyFloat::new(py, x).into_any()),
        Constant::Complex { real, imag } => Ok(PyComplex::from_doubles(py, real, imag).into_any()),
        Constant::Ellipsis => Err(PyErr::new::<PyValueError, _>("ellipsis not supported")),
    }
}

fn realize_list<'py>(py: Python<'py>, list: ExprList) -> PyResult<Bound<'py, PyAny>> {
    let out = PyList::empty(py);

    for elt in list.elts {
        out.append(realize_value(py, elt)?)?;
    }

    Ok(out.into_any())
}

fn realize_dict<'py>(py: Python<'py>, dict: ExprDict) -> PyResult<Bound<'py, PyAny>> {
    let out = PyDict::new(py);

    for (key, val) in dict.keys.into_iter().zip(dict.values.into_iter()) {
        if let Some(key) = key {
            let key = expect_string(key)?;
            let val = realize_value(py, val)?;
            out.set_item(key, val)?;
        } else {
            return Err(PyErr::new::<PyValueError, _>(
                "dictionary splat not supported",
            ));
        }
    }

    Ok(out.into_any())
}

fn expect_string(val: Expr) -> PyResult<String> {
    if let Some(c) = val.constant_expr() {
        match c.value {
            Constant::Str(s) => Ok(s),
            Constant::Bytes(s) => Ok(String::from_utf8(s)?),
            _ => Err(PyErr::new::<PyTypeError, _>(format!(
                "expected string, got {:?}",
                c.kind
            ))),
        }
    } else {
        Err(PyErr::new::<PyValueError, _>("expression is not constant"))
    }
}
