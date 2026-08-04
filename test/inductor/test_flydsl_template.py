# Owner(s): ["module: inductor"]
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel
from torch._inductor.codegen.flydsl.flydsl_scheduling import FlyDSLScheduling
from torch._inductor.codegen.flydsl.flydsl_template import (
    _ordered_unique_input_names,
    FlyDSLTemplate,
)
from torch._inductor.select_algorithm import PartialRender
from torch._inductor.test_case import TestCase
from torch._native.flydsl_utils import _resolve_rocm_arch


class TestFlyDSLTemplate(TestCase):
    def setUp(self):
        super().setUp()
        _resolve_rocm_arch.cache_clear()

    def test_gen_imports(self):
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import flydsl.compiler as flyc", imports)
        self.assertIn("import flydsl.expr as fx", imports)
        self.assertIsInstance(imports, str)

    def test_shared_library_unavailable_reason(self):
        # The package/`flydsl._mlir` probe lives in torch._native.flydsl_utils
        # and is covered by test/python_native/test_flydsl_utils.py; what is
        # Inductor-only, and tested here, is the runtime library check.
        with mock.patch.object(
            flydsl_utils.importlib.util, "find_spec", return_value=None
        ):
            self.assertIn(
                "missing optional dependency",
                flydsl_utils._shared_library_unavailable_reason(),
            )

        with tempfile.TemporaryDirectory() as tmp:
            package_spec = SimpleNamespace(submodule_search_locations=[tmp])
            with mock.patch.object(
                flydsl_utils.importlib.util,
                "find_spec",
                return_value=package_spec,
            ):
                self.assertIn(
                    "runtime shared library",
                    flydsl_utils._shared_library_unavailable_reason(),
                )

                runtime_so = (
                    Path(tmp) / "_mlir" / "_mlir_libs" / "libfly_jit_runtime.so"
                )
                runtime_so.parent.mkdir(parents=True)
                runtime_so.touch()
                ldd = SimpleNamespace(
                    returncode=0,
                    stdout="libdependency.so => not found",
                    stderr="",
                )
                with (
                    mock.patch.object(
                        flydsl_utils.platform, "system", return_value="Linux"
                    ),
                    mock.patch.object(flydsl_utils.subprocess, "run", return_value=ldd),
                ):
                    self.assertIn(
                        "unresolved",
                        flydsl_utils._shared_library_unavailable_reason(),
                    )

    def test_unavailable_runtime_declines_choice(self):
        template_name = f"flydsl_unavailable_test_{id(self)}"
        self.addCleanup(FlyDSLTemplate.all_templates.pop, template_name, None)
        with (
            mock.patch.object(
                FlyDSLTemplate, "_template_from_string", return_value=mock.Mock()
            ),
            mock.patch.object(flydsl_utils, "runtime_available", return_value=False),
        ):
            template = FlyDSLTemplate(name=template_name, source="template")
            choices = []
            result = template.maybe_append_choice(choices)

        self.assertIsInstance(result, NotImplementedError)
        self.assertEqual(choices, [])

    def test_gen_defines(self):
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        defines = kernel.gen_defines(
            TILE_M=128,
            ENABLE_FEATURE=True,
            SCALE=1.5,
        )

        self.assertEqual(
            defines,
            (
                "TILE_M: fx.Constexpr = 128\n"
                "ENABLE_FEATURE: fx.Constexpr = True\n"
                "SCALE: fx.Constexpr = 1.5\n"
            ),
        )

    def test_render_includes_imports(self):
        template = mock.Mock()
        template.render.return_value = (
            "@flyc.kernel\ndef test_kernel_kernel():\n    pass\n"
        )
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        result = kernel.render(template, TILE_M=128)
        code = result.finalize_all()

        self.assertIsInstance(result, PartialRender)
        self.assertTrue(code.lstrip().startswith("import torch"))
        self.assertIn("import flydsl.compiler as flyc", code)
        self.assertIn("@flyc.kernel", code)

    def test_template_env_contains_hooks(self):
        captured_env = {}

        def render(**kwargs):
            captured_env.update(kwargs)
            return "rendered"

        template = mock.Mock()
        template.render = render
        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        kernel.render(template, BLOCK_SIZE=256)

        self.assertEqual(captured_env["kernel_name"], "test_kernel")
        self.assertEqual(captured_env["BLOCK_SIZE"], 256)
        self.assertTrue(callable(captured_env["def_kernel"]))
        self.assertTrue(callable(captured_env["gen_defines"]))
        self.assertTrue(callable(captured_env["get_output"]))

    def test_duplicate_template_name_is_rejected(self):
        template_name = f"flydsl_unique_test_{id(self)}"
        FlyDSLTemplate.all_templates.pop(template_name, None)

        try:
            with mock.patch.object(
                FlyDSLTemplate,
                "_template_from_string",
                return_value=mock.Mock(),
            ):
                FlyDSLTemplate(name=template_name, source="template1")
                FlyDSLTemplate(name=template_name, source="template1")
                with self.assertRaisesRegex(
                    AssertionError, f"duplicate template name, {template_name}"
                ):
                    FlyDSLTemplate(name=template_name, source="template2")
        finally:
            FlyDSLTemplate.all_templates.pop(template_name, None)

    def test_shared_input_buffer_names_are_deduplicated(self):
        shared_names = [bytearray(b"shared").decode() for _ in range(2)]
        self.assertEqual(shared_names[0], shared_names[1])
        self.assertIsNot(shared_names[0], shared_names[1])
        input_nodes = [mock.Mock(), mock.Mock()]
        for input_node, name in zip(input_nodes, shared_names):
            input_node.get_name.return_value = name

        self.assertEqual(
            _ordered_unique_input_names(input_nodes),
            ("shared",),
        )

    def test_scheduling_disables_fusion(self):
        scheduling = FlyDSLScheduling(scheduler=None)
        node1 = mock.Mock()
        node2 = mock.Mock()

        self.assertFalse(scheduling.can_fuse_vertical(node1, node2))
        self.assertFalse(scheduling.can_fuse_horizontal(node1, node2))
        self.assertEqual(scheduling.get_backend_features(device=None), set())

    def test_scheduling_caches_device_arch(self):
        props = mock.Mock(gcnArchName="gfx950:sramecc+:xnack-")
        with (
            mock.patch.dict(
                os.environ,
                {"FLYDSL_GPU_ARCH": "", "HSA_OVERRIDE_GFX_VERSION": ""},
            ),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_properties", return_value=props) as get,
        ):
            self.assertEqual(FlyDSLScheduling._build_flydsl_gpu_arch(0), "gfx950")
            self.assertEqual(FlyDSLScheduling._build_flydsl_gpu_arch(0), "gfx950")
            get.assert_called_once_with(0)

    def test_scheduling_uses_explicit_gpu_arch(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLYDSL_GPU_ARCH": "gfx950:sramecc+:xnack-",
            },
        ):
            self.assertEqual(
                FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0),
                "gfx950",
            )

    def test_scheduling_converts_hsa_override(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLYDSL_GPU_ARCH": "",
                "HSA_OVERRIDE_GFX_VERSION": "9.0.10",
            },
        ):
            self.assertEqual(
                FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0),
                "gfx90a",
            )

    def test_scheduling_preserves_hsa_override_feature_flags(self):
        with mock.patch.dict(
            os.environ,
            {
                "FLYDSL_GPU_ARCH": "",
                "HSA_OVERRIDE_GFX_VERSION": "gfx950:sramecc+",
            },
        ):
            self.assertEqual(
                FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0),
                "gfx950:sramecc+",
            )

    def test_scheduling_returns_none_without_arch_or_device(self):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "FLYDSL_GPU_ARCH": "",
                    "HSA_OVERRIDE_GFX_VERSION": "",
                },
            ),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            self.assertIsNone(FlyDSLScheduling._build_flydsl_gpu_arch(device_index=0))

    def test_precompile_metadata_requires_template_inputs(self):
        scheduling = FlyDSLScheduling(scheduler=None)
        kernel = SimpleNamespace(_template_input_args=[])
        layout = SimpleNamespace(
            size=[1],
            stride=[1],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        self.assertIsNone(
            scheduling._build_precompile_metadata(
                kernel, SimpleNamespace(layout=layout)
            )
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
