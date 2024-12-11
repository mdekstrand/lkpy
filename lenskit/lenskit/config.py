"""
LensKit configuration support.
"""

import os
import warnings
from typing import Sequence

from lenskit.diagnostics import ConfigWarning


def default_torch_device(allowed: Sequence[str] | None = None) -> str:
    """
    Get the default device for PyTorch algorithms.  This is used by algorithms
    that support CUDA or other devices to query whether they are enabled.  See
    :ref:`device-selection` for details.

    Args:
        allowed:
            A list of allowed devices, in case the component is only known to
            support certain devices.

    Returns:
        The device to use.
    """
    import torch

    env = os.environ.get("LK_DEVICE", None)
    if env is not None:
        if env not in ["cpu", "cuda"]:
            warnings.warn(f"configured to use device {env}, not known to work", ConfigWarning)
        if allowed and env not in allowed:
            warnings.warn(f"device {env} disallowed by caller, using anyway", ConfigWarning)
        return env

    if torch.cuda.is_available() and (allowed is None or "cuda" in allowed):
        return "cuda"
    else:
        return "cpu"
