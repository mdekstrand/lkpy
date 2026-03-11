// This file is part of LensKit.
// Copyright (C) 2018-2023 Boise State University.
// Copyright (C) 2023-2026 Drexel University.
// Licensed under the MIT license, see LICENSE.md for details.
// SPDX-License-Identifier: MIT

//! Helper code for the FlexMF models.
use pyo3::prelude::*;

mod convolve;

/// Register the lenskit._accel.flexmf module
pub fn register_flexmf(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let flexmf = PyModule::new(parent.py(), "flexmf")?;
    parent.add_submodule(&flexmf)?;
    flexmf.add_class::<convolve::LightGraph>()?;
    Ok(())
}
