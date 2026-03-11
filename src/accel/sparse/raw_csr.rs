// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! ndarray CSR

/// Raw vector-based CSR structure (no values).
pub struct RawCSR {
    shape: (usize, usize),
    rowptr: Vec<i32>,
    colind: Vec<i32>,
}

/// Build an NDArray CSR.
pub struct RawCSRBuilder {
    shape: (usize, usize),
    row_sizes: Vec<i32>,
}

/// Stage 2 of CSR building.
pub struct RawCSRBuilderStage2 {
    shape: (usize, usize),
    rowptr: Vec<i32>,
    rowcur: Vec<i32>,
    colind: Vec<i32>,
}

impl RawCSR {
    pub fn nnz(&self) -> usize {
        self.colind.len()
    }

    pub fn n_rows(&self) -> usize {
        self.shape.0
    }

    pub fn n_cols(&self) -> usize {
        self.shape.1
    }

    pub fn row_extent(&self, row: i32) -> (usize, usize) {
        let row = row as usize;
        (self.rowptr[row] as usize, self.rowptr[row + 1] as usize)
    }

    pub fn row_cols(&self, row: i32) -> &[i32] {
        let (sp, ep) = self.row_extent(row);
        &self.colind[sp..ep]
    }
}

impl RawCSRBuilder {
    pub fn create(n_rows: usize, n_cols: usize) -> Self {
        RawCSRBuilder {
            shape: (n_rows, n_cols),
            row_sizes: vec![0; n_rows],
        }
    }

    pub fn add_to_rowcount(&mut self, row: i32) {
        self.row_sizes[row as usize] += 1;
    }

    pub fn stage2(self) -> RawCSRBuilderStage2 {
        let (nr, _nc) = self.shape;
        let mut rowptr = Vec::with_capacity(nr + 1);
        let mut nnz = 0;
        rowptr.push(0);
        for i in 0..nr {
            let n = self.row_sizes[i];
            nnz += n as usize;
            rowptr.push(rowptr[i] + n);
        }
        let rowcur = rowptr.clone();
        RawCSRBuilderStage2 {
            shape: self.shape,
            rowptr: rowptr,
            rowcur: rowcur,
            colind: vec![0; nnz],
        }
    }
}

impl RawCSRBuilderStage2 {
    pub fn add_entry(&mut self, row: i32, col: i32) {
        let i = &mut self.rowcur[row as usize];
        self.colind[*i as usize] = col;
        *i += 1;
    }

    pub fn finish(self) -> RawCSR {
        RawCSR {
            shape: self.shape,
            rowptr: self.rowptr,
            colind: self.colind,
        }
    }
}
