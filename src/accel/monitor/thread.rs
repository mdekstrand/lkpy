// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock, Weak};
use std::thread::{self, park_timeout, JoinHandle};
use std::{mem, ptr};
use std::{sync::Arc, time::Duration};

use log::*;
use pyo3::prelude::*;

use crate::monitor::{ActionState, MonitorAction};

type ActionBox = Box<dyn MonitorAction + 'static>;
const UPDATE_TIMEOUT: Duration = Duration::from_millis(100);

static ACTIVE_MONITOR: Mutex<Weak<Monitor>> = Mutex::new(Weak::new());

/// Data tracked by the monitoring thread.
pub(super) struct Monitor {
    thread: OnceLock<JoinHandle<()>>,
    actions: Mutex<Vec<ActionBox>>,
    refcount: AtomicU32,
    cancel_err: AtomicPtr<PyErr>,
}

/// A handle to the monitor, making sure it stays alive.
pub(super) struct MonitorHandle {
    monitor: Arc<Monitor>,
}

impl Monitor {
    fn new() -> Monitor {
        Monitor {
            thread: OnceLock::new(),
            actions: Mutex::new(vec![]),
            refcount: AtomicU32::new(0),
            cancel_err: AtomicPtr::new(ptr::null_mut()),
        }
    }

    pub fn acquire() -> MonitorHandle {
        let mut lock = ACTIVE_MONITOR.lock().expect("monitor thread poisoned");
        let monitor = if let Some(mr) = lock.upgrade() {
            mr
        } else {
            let mon = Arc::new(Monitor::new());
            *lock = Arc::downgrade(&mon);
            mon
        };
        monitor.incr_ref();
        monitor.ensure_running();
        MonitorHandle { monitor }
    }

    fn incr_ref(&self) {
        self.refcount.fetch_add(1, Ordering::Relaxed);
    }

    fn decr_ref(&self) {
        self.refcount.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn take_cancel(&self) -> Option<PyErr> {
        let ptr = self.cancel_err.load(Ordering::Relaxed);
        if ptr.is_null() {
            None
        } else {
            match self.cancel_err.compare_exchange(
                ptr,
                ptr::null_mut(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(p2) => {
                    assert!(p2 == ptr);
                    // SAFETY: we have the only reference to this error.
                    let ptr = unsafe { Box::from_raw(ptr) };
                    Some(*ptr)
                }
                Err(_) => None,
            }
        }
    }

    pub fn ping(&self) {
        if let Some(th) = self.thread.get() {
            th.thread().unpark();
        }
    }

    pub fn add_action<A: MonitorAction + 'static>(&self, action: A) {
        let mut actions = self.actions.lock().expect("monitor poisoned");
        actions.push(Box::new(action));
    }

    fn ensure_running(&self) {
        self.thread.get_or_init(|| {
            thread::Builder::new()
                .name("AccelMonitor".into())
                .spawn(run_monitor)
                .expect("failed to spawn thread")
        });
    }

    fn pump(&self) -> bool {
        let count = self.refcount.load(Ordering::Relaxed);
        if count > 0 {
            Python::attach(|py| {
                if let Err(e) = py.check_signals() {
                    let eptr = ptr::from_mut(Box::leak(Box::new(e)));
                    match self.cancel_err.compare_exchange(
                        ptr::null_mut(),
                        eptr,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => (),
                        Err(_) => unsafe {
                            // SAFETY: we failed to store the pointer, so we are
                            // the owner.

                            // re-box and drop the pointer.
                            let _ = Box::from_raw(eptr);
                        },
                    }
                }
                self.run_actions(Some(py));
            });
            true
        } else {
            self.run_actions(None)
        }
    }

    fn run_actions<'py>(&self, maybe_py: Option<Python<'py>>) -> bool {
        let mut actions = self.actions.lock().expect("monitor thread poisoned");
        let n = actions.len();
        let to_run = mem::replace(&mut *actions, Vec::with_capacity(n));
        let mut to_run = VecDeque::from(to_run);

        run_actions(&mut to_run, &mut *actions, maybe_py);
        actions.len() > 0
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        let ptr = self.cancel_err.load(Ordering::Relaxed);
        if !ptr.is_null() {
            // SAFETY: since we are in drop, no one else has the thread
            let _ptr = unsafe { Box::from_raw(ptr) };
            // drop the pointer
        }
    }
}

impl Deref for MonitorHandle {
    type Target = Monitor;

    fn deref(&self) -> &Self::Target {
        self.monitor.deref()
    }
}

impl AsRef<Monitor> for MonitorHandle {
    fn as_ref(&self) -> &Monitor {
        &self.monitor
    }
}

impl Drop for MonitorHandle {
    fn drop(&mut self) {
        self.monitor.decr_ref();
        self.monitor.ping();
    }
}

/// Send a wakeup signal to the monitor thread, if one is running.
pub(super) fn wakeup_monitor() {
    let lock = ACTIVE_MONITOR.lock().expect("monitor poisoned");
    if let Some(mon) = lock.upgrade() {
        mon.ping();
    }
}

fn run_actions<'py>(
    to_run: &mut VecDeque<ActionBox>,
    out: &mut Vec<ActionBox>,
    py: Option<Python<'py>>,
) {
    while let Some(act) = to_run.pop_front() {
        match act.get_state() {
            ActionState::Finished => (),
            ActionState::Waiting => {
                out.push(act);
            }
            ActionState::Ready => {
                if let Some(py) = py {
                    match act.run_action(py) {
                        Ok(true) => out.push(act),
                        Ok(false) => (),
                        Err(e) => {
                            error!("monitor thread action failed: {}", e);
                        }
                    }
                } else {
                    to_run.push_front(act);
                    return Python::attach(|py| run_actions(to_run, out, Some(py)));
                }
            }
        }
    }
}

fn run_monitor() {
    let monitor = Monitor::acquire();
    Python::attach(|py| {
        py.detach(|| loop {
            park_timeout(UPDATE_TIMEOUT);
            if !monitor.pump() {
                let mut lock = ACTIVE_MONITOR.lock().expect("monitor poisoned");
                *lock = Weak::new();
                break;
            }
        })
    });
}
