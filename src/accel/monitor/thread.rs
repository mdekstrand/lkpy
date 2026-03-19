// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

use std::collections::VecDeque;
use std::mem;
use std::sync::{Mutex, OnceLock, Weak};
use std::thread::{park_timeout, spawn, JoinHandle};
use std::{sync::Arc, time::Duration};

use log::*;
use pyo3::prelude::*;

use crate::monitor::{ActionState, MonitorAction};

type ActionBox = Box<dyn MonitorAction + 'static>;
const UPDATE_TIMEOUT: Duration = Duration::from_millis(200);

static ACTIVE_MONITOR: Mutex<Weak<Monitor>> = Mutex::new(Weak::new());

/// Data tracked by the monitoring thread.
struct Monitor {
    thread: OnceLock<JoinHandle<()>>,
    actions: Mutex<Vec<ActionBox>>,
}

impl Monitor {
    fn new() -> Monitor {
        Monitor {
            thread: OnceLock::new(),
            actions: Mutex::new(vec![]),
        }
    }

    fn acquire() -> Arc<Monitor> {
        let mut lock = ACTIVE_MONITOR.lock().expect("monitor thread poisoned");
        if let Some(tr) = lock.upgrade() {
            tr
        } else {
            let mon = Arc::new(Monitor::new());
            *lock = Arc::downgrade(&mon);
            mon
        }
    }

    fn add_action(&self, action: ActionBox) {
        let mut actions = self.actions.lock().expect("monitor poisoned");
        actions.push(action);
    }

    fn ensure_running(&self) {
        self.thread.get_or_init(|| spawn(run_monitor));
    }

    fn pump(&self) -> bool {
        let mut actions = self.actions.lock().expect("monitor thread poisoned");
        let n = actions.len();
        let to_run = mem::replace(&mut *actions, Vec::with_capacity(n));
        let mut to_run = VecDeque::from(to_run);

        run_actions(&mut to_run, &mut *actions, None);
        actions.len() > 0
    }
}

/// Register a new monitor action with the monitor thread.  If no monitor thread is
/// running, one is started.
pub(super) fn register_monitor<M: MonitorAction + 'static>(action: M) {
    let monitor = Monitor::acquire();

    monitor.add_action(Box::new(action));
    monitor.ensure_running();
}

/// Send a wakeup signal to the monitor thread, if one is running.
pub(super) fn wakeup_monitor() {
    let lock = ACTIVE_MONITOR.lock().expect("monitor poisoned");
    if let Some(mon) = lock.upgrade() {
        if let Some(th) = mon.thread.get() {
            th.thread().unpark();
        }
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
