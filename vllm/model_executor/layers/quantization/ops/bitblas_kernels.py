# SPDX-License-Identifier: Apache-2.0
"""Python binding helpers for BitBLAS INT2×INT8 kernels.

Currently only exposes `matmul_int2_int8` used by the W2A8 linear method.
If BitBLAS is not installed, importing this module will *not* raise –
`matmul_int2_int8` will instead throw a clear RuntimeError the first time it
is called so that model loading can fallback gracefully.
"""
from __future__ import annotations

import types
from typing import Any

try:
    import bitblas  # type: ignore
except Exception as err:  # pragma: no cover – BitBLAS optional.
    _bitblas: Any = None
    _import_err = err  # preserved for error message
else:
    _bitblas = bitblas
    _import_err = None

def _raise():  # noqa: D401
    """Lazy error helper when BitBLAS is unavailable."""
    raise RuntimeError(
        "BitBLAS not found – install `bitblas` to enable `bitblas_w2a8` "
        "quantisation (original import error: %s)" % _import_err)


def _proxy(func_name: str):  # noqa: D401
    """Return thin wrapper that forwards to the matching BitBLAS function."""
    if _bitblas is None:
        return _raise

    if not hasattr(_bitblas, func_name):
        raise RuntimeError(f"Your BitBLAS build does not expose `{func_name}` – "
                           "please rebuild BitBLAS with INT2×INT8 support.")

    _func = getattr(_bitblas, func_name)

    def _wrapper(*args, **kwargs):
        return _func(*args, **kwargs)

    return _wrapper

# Public API ---------------------------------------------------------

matmul_int2_int8 = _proxy("matmul_int2_int8")

__all__ = [
    "matmul_int2_int8",
] 