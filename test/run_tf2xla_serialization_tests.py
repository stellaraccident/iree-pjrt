# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Util functions/classes for jax primitive test harnesses."""

import contextlib
import functools
from typing import Optional
import warnings
import zlib
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax.experimental import jax2tf
from jax.experimental.jax2tf.tests import jax2tf_limitations
from jax.experimental.jax2tf.tests import primitive_harness
import ml_dtypes
import numpy as np
import numpy.random as npr
import tensorflow as tf

_SUPPORTED_DTYPES = [np.float32]


def _harness_matches(harness, group_name, dtype, params):
  if harness.group_name != group_name:
    return False
  if dtype is not None and harness.dtype != dtype:
    return False
  for key, value in params.items():
    if harness.params.get(key, None) != value:
      return False
  return True


_CRASH_LIST_PARMS = []


def _dtype(x):
  if hasattr(x, "dtype"):
    return x.dtype
  elif type(x) in jax.dtypes.python_scalar_dtypes:
    return np.dtype(jax.dtypes.python_scalar_dtypes[type(x)])
  else:
    return np.asarray(x).dtype


def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {np.dtype(key): value for key, value in tol.items()}
  dtype = jax.dtypes.canonicalize_dtype(np.dtype(dtype))
  return tol.get(dtype, _DEFAULT_TOLERANCE[dtype])


@contextlib.contextmanager
def ignore_warning(**kw):
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


def _has_only_supported_dtypes(harness):
  if harness.dtype not in _SUPPORTED_DTYPES:
    return False

  for key, value in harness.params.items():
    if "dtype" in key and value not in _SUPPORTED_DTYPES:
      return False

  return True


def primitives_parameterized(
    harnesses, *, one_containing: Optional[str] = None
):
  """Decorator for tests. This is used to filter the tests.

  Args:
    harnesses: List of Harness objects to be filtered.
    one_containing: If set, only creates one test case for the provided name.

  Returns:
    A parameterized version of the test function with filtered set of harnesses.
  """

  def _filter_harness(harness):
    # TODO(b/295369536) Put a limitations system in place so what's not covered
    # is explicit.
    if not harness.params.get("enable_xla", True):
      return False

    if one_containing is not None and one_containing not in harness.fullname:
      return False

    if not _has_only_supported_dtypes(harness):
      return False

    for crash_item in _CRASH_LIST_PARMS:
      if _harness_matches(
          harness,
          crash_item["group_name"],
          crash_item["dtype"],
          crash_item["params"],
      ):
        return False

    return True

  harnesses = filter(_filter_harness, harnesses)

  return primitive_harness.parameterized(harnesses, include_jax_unimpl=False)


class JaxRunningTestCase(parameterized.TestCase):
  """A test case for JAX conversions."""

  def setUp(self):
    super().setUp()

    # We use the adler32 hash for two reasons.
    # a) it is deterministic run to run, unlike hash() which is randomized.
    # b) it returns values in int32 range, which RandomState requires.
    self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))

  def rng(self):
    return self._rng


class PrimitivesTest(JaxRunningTestCase):

  @primitives_parameterized(
      primitive_harness.all_harnesses,
  )
  @ignore_warning(
      category=UserWarning, message="Using reduced precision for gradient.*"
  )
  def test_prim(self, harness: primitive_harness.Harness):
    device = "iree_cpu"

    def _filter_limitation(limitation):
      return limitation.filter(
          device=device, dtype=harness.dtype, mode="compiled"
      )

    limitations = tuple(
        filter(
            _filter_limitation,
            jax2tf_limitations.Jax2TfLimitation.limitations_for_harness(
                harness
            ),
        )
    )

    func_jax = harness.dyn_fun
    args = harness.dyn_args_maker(self.rng())

    try:
      with jax.jax2tf_associative_scan_reductions(True):
        func_jax(*args)
    except Exception as e:  # pylint: disable=broad-exception-caught
      if "failed to legalize operation 'stablehlo.custom_call'" in str(e):
        logging.warning("Suppressing error %s", e)
      else:
        raise e


if __name__ == "__main__":
  absltest.main()
