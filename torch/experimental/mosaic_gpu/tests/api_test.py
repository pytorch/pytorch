import unittest

from absl.testing import absltest, parameterized
import jax
from jax._src import test_util as jtu
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
import jax.numpy as jnp
import numpy as np
import torch
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


class TestCase(parameterized.TestCase):

    def setUp(self):
        if not HAS_MOSAIC_GPU:
            self.skipTest("jaxlib built without Mosaic GPU")
        if (not jtu.test_device_matches(["cuda"]) or
            not jtu.is_cuda_compute_capability_at_least("9.0")):
            self.skipTest("Only works on GPU with capability >= sm90")
        super().setUp()
        self.prng = np.random.default_rng(1234)
        self.enter_context(jtu.global_config_context(jax_traceback_filtering="off"))
        self.enter_context(mlir.make_ir_context())
        self.enter_context(ir.Location.unknown())


class TorchTest(TestCase):

    def _kernel(self, ctx, i_gmem, o_gmem, _):
        x = mgpu.FragmentedArray.load_strided(i_gmem)
        (x + x).store_untiled(o_gmem)

    def test_eager(self):
        
        ty = jax.ShapeDtypeStruct((128, 128), jnp.float32)
        x = torch.randn((128, 128), dtype=torch.float, device='cuda')
        f = mosaic_gpu.as_torch_gpu_kernel(self._kernel, (1, 1, 1), (128, 1, 1), ty, ty, ())
        y = f(x)
        np.testing.assert_allclose(y.cpu(), x.cpu() * 2)
        del y  # Make sure the destructor runs successfully.

    def test_compile(self):
        ty = jax.ShapeDtypeStruct((128, 128), jnp.float32)
        x = torch.randn((128, 128), dtype=torch.float, device='cuda')
        f = mosaic_gpu.as_torch_gpu_kernel(self._kernel, (1, 1, 1), (128, 1, 1), ty, ty, ())
        compiled_f = torch.compile(f)
        y = compiled_f(x)
        np.testing.assert_allclose(y.cpu(), x.cpu() * 2)
        del y  # Make sure the destructor runs successfully.

    def test_inductor(self):
        pass

if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
