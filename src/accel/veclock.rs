// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2025 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Support for locking vector elements.
//!
//! **HERE BE DRAGONS.**
use std::ops::{Deref, DerefMut};

use parking_lot::lock_api::RawMutex;

/// A vector of locks to go with elements of a vector.
pub struct VecLock<'vec, T>
where
    T: Send + Sync,
{
    data: &'vec mut [T],
    ptr: *mut T,
    locks: Vec<parking_lot::RawMutex>,
}

/// Guard for taking a vector lock.
pub struct VecLockGuard<'lock, 'vec, T>
where
    'vec: 'lock,
    T: Send + Sync,
{
    vlock: &'lock VecLock<'vec, T>,
    index: usize,
}

unsafe impl<'vec, T: Send + Sync> Sync for VecLock<'vec, T> {}

impl<'vec, T: Send + Sync> VecLock<'vec, T> {
    pub fn new(data: &'vec mut [T]) -> VecLock<'vec, T> {
        let ptr = data.as_mut_ptr();
        let mut locks = Vec::with_capacity(data.len());
        for _i in 0..data.len() {
            locks.push(RawMutex::INIT)
        }
        VecLock { data, ptr, locks }
    }

    pub fn lock<'lock>(&'lock self, index: usize) -> VecLockGuard<'lock, 'vec, T>
    where
        'vec: 'lock,
    {
        self.locks[index].lock();
        VecLockGuard { vlock: self, index }
    }
}

impl<'lock, 'vec, T> Drop for VecLockGuard<'lock, 'vec, T>
where
    'vec: 'lock,
    T: Send + Sync,
{
    fn drop(&mut self) {
        unsafe {
            self.vlock.locks[self.index].unlock();
        }
    }
}

impl<'lock, 'vec, T> Deref for VecLockGuard<'lock, 'vec, T>
where
    'vec: 'lock,
    T: Send + Sync,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.vlock.data[self.index]
    }
}

impl<'lock, 'vec, T> DerefMut for VecLockGuard<'lock, 'vec, T>
where
    'vec: 'lock,
    T: Send + Sync,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            let ptr = self.vlock.ptr.add(self.index);
            ptr.as_mut().expect("null pointer")
        }
    }
}
