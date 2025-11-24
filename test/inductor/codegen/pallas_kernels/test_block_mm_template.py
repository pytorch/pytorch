# Owner(s): ["oncall: pt2"]
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from torch._inductor.codegen.pallas_kernels.block_mm_template import (
    PallasTpuBlockMatmulTemplate,
)


"""
class PallasTemplateTests(unittest.TestCase):
    def test_pallas_tpu_block_matmul_template(self):
        # 1. Create an instance of the template
        template = PallasTpuBlockMatmulTemplate(
            name="pallas_tpu_block_mm",
        )

        # 2. Test the src_hash property
        self.assertIsInstance(template.src_hash, str)
        self.assertEqual(len(template.src_hash), 64)

        # 3. Test the get_choice_name method
        choice_name = template.get_choice_name(bm=128, bk=256, bn=512)
        self.assertIsInstance(choice_name, str)

        # 4. Test the render method
        # We need to mock the layout and kwargs for the render method
        class MockLayout:
            size = [512, 512]
        class MockArgs:
            def python_argdefs(self):
                class Arg:
                    def __init__(self, name):
                        self.name = name
                return [Arg("a"), Arg("b"), Arg("c")], None, None, None

        class MockPallasTemplateKernel:
            def __init__(self, *args, **kwargs):
                self.args = MockArgs()
            def _template_from_string(self, s):
                class MockTemplate:
                    def render(self, **kwargs):
                        return ""
                return MockTemplate()
            def render(self):
                return ""

        kernel = MockPallasTemplateKernel()
        rendered_code = template.get_make_kernel_render("test_choice", MockLayout(), bm=128, bk=256, bn=512)(kernel)[1]()
        self.assertIsInstance(rendered_code, str)

        # Check for imports
        self.assertIn("import jax", rendered_code)
        self.assertIn("from jax.experimental import pallas as pl", rendered_code)
"""


class MockLayout:
    def __init__(self, dtype) -> None:
        self.dtype = dtype


class MockInputNode:
    def __init__(self, dtype) -> None:
        self._dtype = dtype

    def get_layout(self) -> MockLayout:
        return MockLayout(self._dtype)


class PallasKernelCorrectnessJaxTests(unittest.TestCase):
    def test_block_mm_template(self):
        dtypes_to_test = [
            # jnp.int8,
            # jnp.int16,
            # jnp.int32,
            # jnp.float8_e5m2,
            # jnp.float8_e4m3fn,
            # jnp.float8_e4m3b11fnuz,
            jnp.bfloat16,
            jnp.float32,
        ]

        for x_dtype in dtypes_to_test:
            for y_dtype in dtypes_to_test:
                with self.subTest(x_dtype=x_dtype, y_dtype=y_dtype):
                    choice_name = f"test_kernel_{x_dtype.__name__}_{y_dtype.__name__}"

                    mock_input_nodes = [MockInputNode(x_dtype), MockInputNode(y_dtype)]
                    kwargs = {"bm": 128, "bk": 128, "bn": 128}

                    kernel_code = PallasTpuBlockMatmulTemplate._render_kernel(
                        mock_input_nodes
                    ).replace("<KERNEL_NAME>", choice_name)
                    pallas_call_code = PallasTpuBlockMatmulTemplate._render_pallas_call(
                        kwargs
                    ).replace("<KERNEL_NAME>", choice_name)

                    scope = {
                        "jax": jax,
                        "jnp": jnp,
                        "pl": pl,
                        "pltpu": pltpu,
                    }
                    exec(kernel_code + "\n" + pallas_call_code, scope)

                    pallas_call_for_test = scope[f"{choice_name}_pallas_call"]

                    k1, k2 = jax.random.split(jax.random.key(0))

                if jnp.issubdtype(x_dtype, jnp.integer):
                    x = jax.random.randint(k1, (512, 512), 0, 10, dtype=x_dtype)
                else:
                    x = jax.random.uniform(k1, (512, 512), dtype=x_dtype)

                if jnp.issubdtype(y_dtype, jnp.integer):
                    y = jax.random.randint(k2, (512, 512), 0, 10, dtype=y_dtype)
                else:
                    y = jax.random.uniform(k2, (512, 512), dtype=y_dtype)

                z = pallas_call_for_test(x, y)

                if x_dtype == jnp.bfloat16 or y_dtype == jnp.bfloat16:
                    rtol = 1e-2
                    atol = 1e-2
                elif jnp.issubdtype(x_dtype, jnp.integer) and jnp.issubdtype(
                    y_dtype, jnp.integer
                ):
                    rtol = 0
                    atol = 0
                else:
                    rtol = 1e-6
                    atol = 1e-6

                np.testing.assert_allclose(z, jnp.dot(x, y), rtol=rtol, atol=atol)


from torch._inductor.test_case import run_tests


if __name__ == "__main__":
    run_tests()
