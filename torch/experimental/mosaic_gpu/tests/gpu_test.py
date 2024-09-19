# Copyright 2024 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Mosaic GPU DSL functions and utilities."""

import enum
import unittest

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
import jax.numpy as jnp
import numpy as np
try:
    import jax._src.lib.mosaic_gpu  # noqa: F401
    HAS_MOSAIC_GPU = True
except ImportError:
    HAS_MOSAIC_GPU = False

    class Dimension(enum.IntEnum):  # Just to make parameterized tests expand ok
        x = 0
        y = 1
        z = 2
else:
    from jax.experimental.mosaic import gpu as mosaic_gpu
    from jax.experimental.mosaic.gpu import dsl as mgpu
    from jax.experimental.mosaic.gpu import profiler
    from jax.experimental.mosaic.gpu.utils import *  # noqa: F403
    from jax._src.lib.mlir.dialects import gpu
    Dimension = gpu.Dimension


class TorchTest(TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Test requires PyTorch")
            cls.torch = torch

    def test_basic(self):
        def kernel(ctx, i_gmem, o_gmem, _):
            x = mgpu.FragmentedArray.load_strided(i_gmem)
            (x + x).store_untiled(o_gmem)

        ty = jax.ShapeDtypeStruct((128, 128), jnp.float32)
        x = self.torch.randn((128, 128), dtype=self.torch.float, device='cuda')
        f = mosaic_gpu.as_torch_gpu_kernel(kernel, (1, 1, 1), (128, 1, 1), ty, ty, ())
        y = f(x)
        np.testing.assert_allclose(y.cpu(), x.cpu() * 2)
        del y  # Make sure the destructor runs successfully.


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
