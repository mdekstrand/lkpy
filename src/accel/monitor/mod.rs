// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use pyo3::{PyResult, Python};

pub mod cancel;
pub mod progress;
mod thread;

pub use cancel::SigCancel;
pub use cancel::WithCancel;

#[derive(Clone, Copy, Eq, PartialEq)]
enum ActionState {
    Ready,
    Waiting,
    Finished,
}

/// Trait for actions to register with the thread monitor.
trait MonitorAction: Send {
    /// Check if this monitor needs to run.  This allows the monitor thread to
    /// avoid reacquiring the GIL when no monitors need action.
    ///
    /// **Note:** the action may be run even if it reports that it does not need
    /// to be run.
    fn get_state(&self) -> ActionState;

    /// Run this thread's monitor action.  Should return `true` if the monitor
    /// task should remain active, and `false` to unregister it.
    fn run_action<'py>(&self, py: Python<'py>) -> PyResult<bool>;
}
