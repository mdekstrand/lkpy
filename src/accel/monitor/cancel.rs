// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for cancelling operations.

use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use pyo3::{exceptions::asyncio::CancelledError, PyErr, PyResult};
use rayon::iter::{
    plumbing::{Consumer, Folder, ProducerCallback, Reducer, UnindexedConsumer},
    IndexedParallelIterator, ParallelIterator,
};

use crate::monitor::thread::{Monitor, MonitorHandle};

/// Handle for handling signal-based cancellation (e.g. `KeyboardInterrupt`).
#[derive(Clone)]
pub struct SigCancel {
    monitor: MonitorHandle,
    cancel_flag: Arc<AtomicBool>,
}

pub trait WithCancel {
    type Cancelable;

    fn with_sig_cancellation(self) -> Self::Cancelable;
}

impl SigCancel {
    pub fn acquire() -> SigCancel {
        let monitor = Monitor::acquire();
        SigCancel {
            monitor,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Query if this canceller has been cancelled. Does *not* check for new errors.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }

    pub fn check_cancel(&self) -> PyResult<()> {
        self.ok_or_cancelled(())
    }

    pub fn ok_or_cancelled<T>(&self, val: T) -> PyResult<T> {
        if let Some(err) = self.monitor.take_cancel() {
            self.cancel_flag.store(true, Ordering::Relaxed);
            Err(err)
        } else {
            Ok(val)
        }
    }

    pub fn wrap_iter<I: ParallelIterator>(
        &self,
        iter: I,
    ) -> impl ParallelIterator<Item = PyResult<I::Item>> {
        let cancel = self.clone();
        iter.map(move |i| cancel.ok_or_cancelled(i))
    }
}

pub struct CancelParIter<I>
where
    I: ParallelIterator,
{
    base: I,
}

impl<I: ParallelIterator> ParallelIterator for CancelParIter<I> {
    type Item = I::Item;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        todo!()
    }
}

pub struct CancelConsumer<T, C>
where
    T: Send,
    C: Consumer<T>,
{
    base: C,
    cancel: SigCancel,
    _phantom: PhantomData<T>,
}

pub struct CancelFolder<T, F>
where
    T: Send,
    F: Folder<T>,
{
    fold: PyResult<F>,
    cancel: SigCancel,
    _phantom: PhantomData<T>,
}

pub struct CancelReducer<T, R>
where
    T: Send,
    R: Reducer<T>,
{
    base: R,
    cancel: SigCancel,
    _phantom: PhantomData<T>,
}

impl<T, C> Consumer<T> for CancelConsumer<T, C>
where
    T: Send,
    C: Consumer<T>,
{
    type Folder = CancelFolder<T, C::Folder>;

    type Reducer = CancelReducer<C::Result, C::Reducer>;

    type Result = PyResult<C::Result>;

    fn split_at(self, index: usize) -> (Self, Self, Self::Reducer) {
        let (left, right, combine) = self.base.split_at(index);
        let left = CancelConsumer {
            base: left,
            cancel: self.cancel.clone(),
            _phantom: PhantomData,
        };
        let right = CancelConsumer {
            base: right,
            cancel: self.cancel.clone(),
            _phantom: PhantomData,
        };
        let combine = CancelReducer {
            base: combine,
            cancel: self.cancel.clone(),
            _phantom: PhantomData,
        };
        (left, right, combine)
    }

    fn into_folder(self) -> Self::Folder {
        CancelFolder {
            fold: Ok(self.base.into_folder()),
            cancel: self.cancel,
            _phantom: PhantomData,
        }
    }

    fn full(&self) -> bool {
        self.base.full() || self.cancel.is_cancelled()
    }
}

impl<T, C> UnindexedConsumer<T> for CancelConsumer<T, C>
where
    T: Send,
    C: UnindexedConsumer<T>,
{
    fn split_off_left(&self) -> Self {
        CancelConsumer {
            base: self.base.split_off_left(),
            cancel: self.cancel.clone(),
            _phantom: PhantomData,
        }
    }

    fn to_reducer(&self) -> Self::Reducer {
        CancelReducer {
            base: self.base.to_reducer(),
            cancel: self.cancel.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T, F> Folder<T> for CancelFolder<T, F>
where
    T: Send,
    F: Folder<T>,
{
    type Result = PyResult<F::Result>;

    fn consume(self, item: T) -> Self {
        let fold = self.fold.map(|f| f.consume(item));
        CancelFolder {
            fold,
            cancel: self.cancel,
            _phantom: PhantomData,
        }
    }

    fn complete(self) -> Self::Result {
        self.fold.map(|f| f.complete())
    }

    fn full(&self) -> bool {
        match &self.fold {
            Ok(f) => f.full() || self.cancel.is_cancelled(),
            Err(_e) => true,
        }
    }
}

impl<T, R> Reducer<PyResult<T>> for CancelReducer<T, R>
where
    T: Send,
    R: Reducer<T>,
{
    fn reduce(self, left: PyResult<T>, right: PyResult<T>) -> PyResult<T> {
        if let Err(err) = self.cancel.check_cancel() {
            Err(err)
        } else {
            match (left, right) {
                (Ok(l), Ok(r)) => Ok(self.base.reduce(l, r)),
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
            }
        }
    }
}
