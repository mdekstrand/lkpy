// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use pyo3::{intern, prelude::*, types::PyDict};

use crate::monitor::thread::{wakeup_monitor, Monitor};
use crate::monitor::{ActionState, MonitorAction};

struct ProgressData {
    pb: Py<PyAny>,
    count: AtomicUsize,
}

struct ProgressMonitor {
    data: Weak<ProgressData>,
    last_count: Cell<usize>,
}

/// Thin Rust wrapper around a LensKit progress bar.
///
/// This method applies internal throttling to reduce the number of calls
/// to the Python progress bar.
pub(crate) struct ProgressHandle {
    data: Option<Arc<ProgressData>>,
}

impl ProgressHandle {
    pub fn from_input<'py>(maybe_pb: Bound<'py, PyAny>) -> Self {
        let pb = if maybe_pb.is_none() {
            None
        } else {
            Some(maybe_pb.unbind())
        };
        Self::new(pb)
    }

    pub fn new(pb: Option<Py<PyAny>>) -> Self {
        pb.map(|pb| {
            let data = Arc::new(ProgressData {
                pb,
                count: AtomicUsize::new(0),
            });
            let monitor = ProgressMonitor {
                data: Arc::downgrade(&data),
                last_count: Cell::new(0),
            };
            Monitor::acquire().add_action(monitor);

            ProgressHandle { data: Some(data) }
        })
        .unwrap_or_else(Self::null)
    }

    pub fn null() -> Self {
        ProgressHandle { data: None }
    }

    pub fn tick(&self) {
        self.advance(1);
    }

    pub fn advance(&self, n: usize) {
        if let Some(data) = &self.data {
            data.count.fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Force an update of the progress bar.
    pub fn flush(&self) {
        wakeup_monitor();
    }
}

impl Clone for ProgressHandle {
    fn clone(&self) -> Self {
        ProgressHandle {
            data: self.data.clone(),
        }
    }
}

impl Drop for ProgressHandle {
    fn drop(&mut self) {
        wakeup_monitor();
    }
}

impl MonitorAction for ProgressMonitor {
    fn get_state(&self) -> ActionState {
        if let Some(ptr) = self.data.upgrade() {
            let count = ptr.count.load(Ordering::Relaxed);
            if count > self.last_count.get() {
                ActionState::Ready
            } else {
                ActionState::Waiting
            }
        } else {
            ActionState::Finished
        }
    }

    fn run_action<'py>(&self, py: Python<'py>) -> PyResult<bool> {
        let ptr = if let Some(ptr) = self.data.upgrade() {
            ptr
        } else {
            return Ok(false);
        };

        let count = ptr.count.load(Ordering::Relaxed);

        if count > self.last_count.get() {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "completed"), count)?;
            ptr.pb
                .call_method(py, intern!(py, "update"), (), Some(&kwargs))?;
            self.last_count.set(count);
        }

        Ok(true)
    }
}
