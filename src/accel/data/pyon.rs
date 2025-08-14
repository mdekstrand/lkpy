// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for reading invalid JSON that is actually valid Python expression syntax.

use std::any::Any;

use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString},
};

use python_parser::{ast::SetItem, eval_input};
use python_parser::{
    ast::{DictItem, Expression},
    make_strspan,
};

/// Parse a “pyon” object.
#[pyfunction]
pub fn pyon_loads<'py>(py: Python<'py>, text: &str) -> PyResult<PyObject> {
    let (_r, exprs) = eval_input(make_strspan(text))
        .map_err(|_e| PyErr::new::<PyValueError, _>("Python parse error"))?;
    if exprs.len() != 1 {
        Err(PyValueError::new_err(format!(
            "parsed {} expressions, expected 1",
            exprs.len()
        )))
    } else {
        let obj = realize_value(py, exprs.into_iter().next().unwrap())?;
        Ok(obj.unbind())
    }
}

fn realize_value<'py>(py: Python<'py>, ast: Expression) -> PyResult<Bound<'py, PyAny>> {
    match ast {
        Expression::None => Ok(PyNone::get(py).to_owned().into_any()),
        Expression::True => Ok(PyBool::new(py, true).to_owned().into_any()),
        Expression::False => Ok(PyBool::new(py, false).to_owned().into_any()),
        Expression::Bytes(val) => Ok(PyString::new(py, str::from_utf8(&val)?).into_any()),
        Expression::String(chunks) => {
            let string: String = chunks.into_iter().map(|c| c.content).collect();
            Ok(PyString::new(py, &string).into_any())
        }
        Expression::Int(n) => Ok(PyInt::new(py, n).into_any()),
        Expression::Float(x) => Ok(PyFloat::new(py, x).into_any()),
        Expression::Ellipsis => Err(PyErr::new::<PyValueError, _>("ellipsis not supported")),
        Expression::ListLiteral(items) => {
            let out = PyList::empty(py);

            for elt in items {
                if let SetItem::Unique(e) = elt {
                    out.append(realize_value(py, e)?)?;
                } else {
                    return Err(PyValueError::new_err("splat not supported"));
                }
            }

            Ok(out.into_any())
        }
        Expression::DictLiteral(items) => {
            let out = PyDict::new(py);

            for elt in items {
                if let DictItem::Unique(name, value) = elt {
                    let name = expect_string(name)?;
                    out.set_item(name, realize_value(py, value)?)?;
                } else {
                    return Err(PyValueError::new_err("splat not supported"));
                }
            }

            Ok(out.into_any())
        }
        _ => Err(PyErr::new::<PyValueError, _>(format!(
            "unsupported expression type {:?}",
            ast.type_id()
        ))),
    }
}

fn expect_string(val: Expression) -> PyResult<String> {
    match val {
        Expression::String(s) => Ok(s.into_iter().map(|c| c.content).collect()),
        Expression::Bytes(s) => Ok(String::from_utf8(s)?),
        _ => Err(PyErr::new::<PyTypeError, _>(format!(
            "expected string, got {:?}",
            val
        ))),
    }
}
