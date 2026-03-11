// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Accelerate LightGCN convolution layers.
use arrow::{
    array::{make_array, ArrayData, Int32Array},
    pyarrow::PyArrowType,
};
use log::*;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;

use crate::{
    arrow::checked_array,
    check_pyvalue,
    sparse::{RawCSR, RawCSRBuilder},
};

/// Neighborhood graph for LightGCN.
#[pyclass]
pub struct LightGraph {
    user_edges: RawCSR,
    item_edges: RawCSR,
    user_norms: Array1<f32>,
    item_norms: Array1<f32>,
}

#[pymethods]
impl LightGraph {
    /// Create a new neghbor matrix.
    #[new]
    fn new(
        n_users: usize,
        user_nums: PyArrowType<ArrayData>,
        n_items: usize,
        item_nums: PyArrowType<ArrayData>,
    ) -> PyResult<LightGraph> {
        let user_nums: Int32Array = checked_array("user numbers", &make_array(user_nums.0))?;
        let item_nums: Int32Array = checked_array("item numbers", &make_array(item_nums.0))?;

        let uslice = user_nums.values();
        let islice = item_nums.values();
        let n = uslice.len();
        check_pyvalue!(islice.len() == n, "user-item number mismatch");

        debug!("accumulating user-item rows");
        let mut ue_builder = RawCSRBuilder::create(n_users, n_items);
        let mut ie_builder = RawCSRBuilder::create(n_items, n_users);
        for i in 0..n {
            ue_builder.add_to_rowcount(uslice[i]);
            ie_builder.add_to_rowcount(islice[i]);
        }

        let mut ue_builder = ue_builder.stage2();
        let mut ie_builder = ie_builder.stage2();
        let mut user_norms = Array1::zeros(n_users);
        let mut item_norms = Array1::zeros(n_items);

        for i in 0..n {
            let user = uslice[i];
            let item = islice[i];
            ue_builder.add_entry(user, item);
            ie_builder.add_entry(item, user);
            user_norms[user as usize] += 1.0;
            item_norms[item as usize] += 1.0;
        }

        let user_norms = user_norms.sqrt().recip();
        let item_norms = item_norms.sqrt().recip();

        let conv = LightGraph {
            user_edges: ue_builder.finish(),
            item_edges: ie_builder.finish(),
            user_norms,
            item_norms,
        };

        Ok(conv)
    }

    fn user_layer_matrices<'py>(
        &self,
        py: Python<'py>,
        users: Bound<'py, PyArray1<i32>>,
        n_layers: u8,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
        let py_users = users.readonly();
        let users = py_users.as_array();
        let mut acc = make_accumulator(
            n_layers,
            users.len(),
            &self.user_edges,
            &self.item_edges,
            &self.user_norms,
            &self.item_norms,
        );
        for user in users {
            let user = *user;
            acc.acc_nbrs(user as usize, &[user]);
        }

        let mut out = Vec::with_capacity(n_layers as usize);
        acc.finalize(py, &mut out);
        Ok(out)
    }

    fn item_layer_matrices<'py>(
        &self,
        py: Python<'py>,
        items: Bound<'py, PyArray1<i32>>,
        n_layers: u8,
    ) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
        let py_items = items.readonly();
        let items = py_items.as_array();
        let mut acc = make_accumulator(
            n_layers,
            items.len(),
            &self.item_edges,
            &self.user_edges,
            &self.item_norms,
            &self.user_norms,
        );
        for item in items {
            let item = *item;
            acc.acc_nbrs(item as usize, &[item]);
        }

        let mut out = Vec::with_capacity(n_layers as usize);
        acc.finalize(py, &mut out);
        Ok(out)
    }
}

fn make_accumulator<'a>(
    layers: u8,
    n_rows: usize,
    m1: &'a RawCSR,
    m2: &'a RawCSR,
    n1: &'a Array1<f32>,
    n2: &'a Array1<f32>,
) -> NbrAcc<'a> {
    assert!(layers >= 1);
    if layers == 1 {
        NbrAcc::create_leaf(n_rows, m1, n1, n2)
    } else {
        NbrAcc::create_chained(
            n_rows,
            m1,
            n1,
            n2,
            // next one uses matrices in opposite order
            Some(make_accumulator(layers - 1, n_rows, m2, m1, n2, n1)),
        )
    }
}

struct NbrAcc<'a> {
    nbr_mat: &'a RawCSR,
    row_norms: &'a Array1<f32>,
    col_norms: &'a Array1<f32>,
    out_mat: Array2<f32>,
    col_counts: Array1<i32>,
    next: Option<Box<NbrAcc<'a>>>,
}

impl<'a> NbrAcc<'a> {
    fn create_chained(
        n_rows: usize,
        nbr_mat: &'a RawCSR,
        row_norms: &'a Array1<f32>,
        col_norms: &'a Array1<f32>,
        next: Option<NbrAcc<'a>>,
    ) -> Self {
        NbrAcc {
            nbr_mat,
            row_norms,
            col_norms,
            out_mat: Array2::zeros((n_rows, nbr_mat.n_cols())),
            col_counts: Array1::zeros(nbr_mat.n_cols()),
            next: next.map(Box::new),
        }
    }

    fn create_leaf(
        n_rows: usize,
        nbr_mat: &'a RawCSR,
        row_norms: &'a Array1<f32>,
        col_norms: &'a Array1<f32>,
    ) -> Self {
        Self::create_chained(n_rows, nbr_mat, row_norms, col_norms, None)
    }

    fn acc_nbrs(&mut self, out_row: usize, rows: &[i32]) {
        for j in rows {
            let j = *j;
            let jv = self.row_norms[j as usize];
            let nbrs = self.nbr_mat.row_cols(j);
            for n in nbrs {
                let n = *n as usize;
                self.out_mat[(out_row, n)] += jv * self.col_norms[n];
                self.col_counts[n] += 1;
            }
            if let Some(next) = &mut self.next {
                next.acc_nbrs(out_row, nbrs);
            }
        }
    }

    fn finalize<'py>(self, py: Python<'py>, output: &mut Vec<Bound<'py, PyArray2<f32>>>) {
        let arr = self.out_mat.to_pyarray(py);
        output.push(arr);
        if let Some(next) = self.next {
            next.finalize(py, output);
        }
    }
}
