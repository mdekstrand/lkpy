// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Accumulator for scores in k-NN.
use std::collections::BinaryHeap;

use arrow::array::{Float32Array, Float32Builder, Int32Array, Int32Builder};
use ordered_float::NotNan;
use parking_lot::Mutex;
use parking_lot::MutexGuard;
use pyo3::{exceptions::PyValueError, prelude::*};

/// Thread-safe accumulator for scores.
pub(super) struct ScoreAccumulator<T> {
    limit: u16,
    data: parking_lot::Mutex<AccStore<T>>,
}

#[derive(Clone)]
enum AccStore<T> {
    Empty,
    Partial(Vec<AccEntry<T>>),
    Full(BinaryHeap<AccEntry<T>>),
}

impl<T> Default for AccStore<T> {
    fn default() -> Self {
        AccStore::Empty
    }
}

impl ScoreAccumulator<()> {
    pub fn add_weight(&self, weight: f32) -> PyResult<()> {
        self.add_value(weight, ())
    }
}

impl<T: Clone> ScoreAccumulator<T> {
    pub fn new_array(n: usize, active: &Int32Array, limit: u16) -> Vec<ScoreAccumulator<T>> {
        // create accumulators for all items, and enable the targets
        let mut heaps: Vec<ScoreAccumulator<T>> = Vec::with_capacity(n);
        for _i in 0..n {
            heaps.push(Self::disabled());
        }
        for i in active.iter() {
            if let Some(i) = i {
                heaps[i as usize].enable(limit)
            }
        }
        heaps
    }
}

impl<T> ScoreAccumulator<T> {
    /// Create a disabled score accumulator.
    pub fn disabled() -> Self {
        ScoreAccumulator {
            limit: 0,
            data: Mutex::new(AccStore::Empty),
        }
    }

    /// Enable a score accumulator.
    pub fn enable(&mut self, limit: u16) {
        self.limit = limit;
    }

    pub fn enabled(&self) -> bool {
        self.limit > 0
    }

    pub fn len(&self) -> usize {
        let data = self.data.lock();
        match &*data {
            AccStore::Empty => 0,
            AccStore::Partial(v) => v.len(),
            AccStore::Full(h) => h.len(),
        }
    }

    pub fn add_value(&self, weight: f32, value: T) -> PyResult<()> {
        if !self.enabled() {
            return Ok(());
        }

        let mut data = self.storage_lock();
        let entry = AccEntry::new(weight, value)?;
        if let Some(vec) = data.vector_mut(self.limit as usize) {
            vec.push(entry);
        } else {
            let heap = data.heap_mut(self.limit as usize);
            if entry.weight > heap.peek().unwrap().weight {
                heap.push(entry);
                while heap.len() > self.limit as usize {
                    heap.pop();
                }
            }
        }

        Ok(())
    }

    /// Get the underlying storage, locked.
    fn storage_lock<'lock, 'a: 'lock>(&'a self) -> MutexGuard<'lock, AccStore<T>> {
        self.data.lock()
    }

    /// Get the underlying storage as a direct mutable reference.
    fn storage_mut(&mut self) -> &mut AccStore<T> {
        self.data.get_mut()
    }
}

impl<T> AccStore<T> {
    fn heap_mut(&mut self, limit: usize) -> &mut BinaryHeap<AccEntry<T>> {
        match self {
            AccStore::Full(h) => h,
            AccStore::Empty => {
                let heap = BinaryHeap::with_capacity(limit + 1);
                *self = AccStore::Full(heap);
                self.heap_mut(limit)
            }
            AccStore::Partial(vec) => {
                let mut heap = BinaryHeap::with_capacity(limit + 1);
                while let Some(v) = vec.pop() {
                    heap.push(v);
                }
                *self = AccStore::Full(heap);
                self.heap_mut(limit)
            }
        }
    }

    fn vector_mut(&mut self, limit: usize) -> Option<&mut Vec<AccEntry<T>>> {
        match self {
            AccStore::Empty => {
                let vec = Vec::with_capacity(limit);
                *self = AccStore::Partial(vec);
                self.vector_mut(limit)
            }
            AccStore::Partial(v) if v.len() < limit => Some(v),
            _ => None,
        }
    }

    pub fn total_weight(&self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Full(heap) => heap.iter().map(AccEntry::get_weight).sum(),
            Self::Partial(vec) => vec.iter().map(AccEntry::get_weight).sum(),
        }
    }
}

impl AccStore<f32> {
    pub fn weighted_sum(&self) -> f32 {
        match self {
            Self::Empty => 0.0,
            Self::Full(heap) => heap.iter().map(|a| a.weight * a.data).sum(),
            Self::Partial(vec) => vec.iter().map(|a| a.weight * a.data).sum(),
        }
    }
}

/// Entries in the accumulator heaps.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct AccEntry<T> {
    weight: NotNan<f32>,
    data: T,
}

impl<T> AccEntry<T> {
    fn new(weight: f32, payload: T) -> PyResult<AccEntry<T>> {
        Ok(AccEntry {
            weight: NotNan::new(weight)
                .map_err(|_e| PyValueError::new_err("similarity is null"))?,
            data: payload,
        })
    }

    fn get_weight(&self) -> f32 {
        self.weight.into_inner()
    }
}

impl<T> PartialEq for AccEntry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.weight == other.weight
    }
}

impl<T> Eq for AccEntry<T> {}

impl<T> PartialOrd for AccEntry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // reverse the ordering to make a min-heap
        other.weight.partial_cmp(&self.weight)
    }
}

impl<T> Ord for AccEntry<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reverse the ordering to make a min-heap
        other.weight.cmp(&self.weight)
    }
}
pub(super) fn collect_items_counts<T>(
    heaps: &[ScoreAccumulator<T>],
    tgt_is: &Int32Array,
) -> Int32Array {
    let mut out = Int32Builder::with_capacity(tgt_is.len());
    for ti in tgt_is {
        if let Some(ti) = ti {
            let acc = &heaps[ti as usize];
            out.append_value(acc.len() as i32);
        } else {
            out.append_null();
        }
    }
    out.finish()
}

pub(super) fn collect_items_averaged(
    heaps: &mut [ScoreAccumulator<f32>],
    tgt_is: &Int32Array,
    min_nbrs: usize,
) -> Float32Array {
    let mut out = Float32Builder::with_capacity(tgt_is.len());
    for ti in tgt_is {
        if let Some(ti) = ti {
            let acc = &mut heaps[ti as usize];
            if acc.len() >= min_nbrs {
                let store = acc.storage_mut();
                let score = store.weighted_sum() / store.total_weight();
                out.append_value(score);
            } else {
                out.append_null();
            }
        } else {
            out.append_null();
        }
    }
    out.finish()
}

pub(super) fn collect_items_summed(
    heaps: &mut [ScoreAccumulator<()>],
    tgt_is: &Int32Array,
    min_nbrs: usize,
) -> Float32Array {
    let mut out = Float32Builder::with_capacity(tgt_is.len());
    for ti in tgt_is {
        if let Some(ti) = ti {
            let acc = &mut heaps[ti as usize];
            if acc.len() >= min_nbrs {
                let store = acc.storage_mut();
                let score = store.total_weight();
                out.append_value(score);
            } else {
                out.append_null();
            }
        } else {
            out.append_null();
        }
    }
    out.finish()
}
