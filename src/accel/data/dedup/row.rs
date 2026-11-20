// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::HashSet;

use arrow::array::{BooleanArray, BooleanBuilder};
use arrow::error::Result as ArrowResult;
use arrow::{
    datatypes::Schema,
    row::{RowConverter, SortField},
};
use xxhash_rust::xxh64::xxh64;

use crate::data::dedup::DedupMemory;

pub(super) struct RowMemory {
    converter: RowConverter,
    table: HashSet<u64>,
}

impl RowMemory {
    pub fn new(schema: &Schema) -> ArrowResult<Self> {
        let mut fields = vec![];
        for field in schema.fields() {
            fields.push(SortField::new(field.data_type().clone()))
        }
        Ok(RowMemory {
            converter: RowConverter::new(fields)?,
            table: HashSet::with_capacity(100_000),
        })
    }
}

impl DedupMemory for RowMemory {
    fn len(&self) -> usize {
        self.table.len()
    }

    fn batch_unique_mask(
        &mut self,
        batch: &arrow::array::RecordBatch,
    ) -> ArrowResult<BooleanArray> {
        let cols = batch.columns();
        let mut result = BooleanBuilder::with_capacity(batch.num_rows());
        for row in self.converter.convert_columns(cols)?.into_iter() {
            let hash = xxh64(row.data(), 42);
            result.append_value(self.table.insert(hash))
        }
        Ok(result.finish())
    }
}
