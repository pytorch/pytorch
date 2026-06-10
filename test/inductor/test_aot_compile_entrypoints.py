# Owner(s): ["module: inductor"]

import os
import tempfile
from unittest.mock import patch

import torch
import torch._inductor
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import run_tests


class TestAOTCompileEntrypoints(TestCase):
    def test_aot_compile_dispatches_to_compile_aot_core(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        args = (torch.ones(2),)
        gm = torch.export.export(Model(), args, strict=True).module()
        options = {"aot_inductor.package": True}
        sentinel = object()
        calls = []

        def fake_compile_aot_core(gm_arg, args_arg, kwargs_arg, *, options=None):
            calls.append((gm_arg, args_arg, kwargs_arg, options))
            return sentinel

        with patch(
            "torch._inductor._compile_aot_core",
            side_effect=fake_compile_aot_core,
        ):
            result = torch._inductor.aot_compile(gm, args, options=options)

        self.assertIs(result, sentinel)
        self.assertEqual(len(calls), 1)
        gm_arg, args_arg, kwargs_arg, options_arg = calls[0]
        self.assertIs(gm_arg, gm)
        self.assertIs(args_arg, args)
        self.assertIsNone(kwargs_arg)
        self.assertIs(options_arg, options)

    def test_aoti_compile_and_package_dispatches_to_compile_aot_core(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        args = (torch.ones(2),)
        exported_program = torch.export.export(Model(), args, strict=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = os.path.join(temp_dir, "model.pt2")
            so_path = os.path.join(temp_dir, "model.so")
            options = {"max_autotune": False}
            expected_exported_program = exported_program
            expected_package_path = package_path
            compile_aot_core_calls = []
            package_aoti_calls = []

            def fake_compile_aot_core(gm, args, kwargs, *, options=None):
                compile_aot_core_calls.append((gm, args, kwargs, options))
                return [so_path]

            def fake_minifier_wrapper(
                func,
                exported_program,
                *,
                inductor_configs,
                package_path=None,
            ):
                self.assertIs(exported_program, expected_exported_program)
                self.assertIs(inductor_configs, options)
                self.assertEqual(package_path, expected_package_path)
                gm = exported_program.module(check_guards=False)
                args, kwargs = exported_program.example_inputs
                return func(
                    gm,
                    args,
                    kwargs,
                    inductor_configs=inductor_configs,
                    package_path=package_path,
                )

            def fake_package_aoti(path, files):
                package_aoti_calls.append((path, files))
                return path

            with (
                patch(
                    "torch._inductor._compile_aot_core",
                    side_effect=fake_compile_aot_core,
                ),
                patch(
                    "torch._inductor.aot_compile",
                    # Tripwire: the package entry point should use the shared
                    # core directly instead of recursing through aot_compile().
                    side_effect=AssertionError(
                        "package entry point called aot_compile"
                    ),
                ),
                patch(
                    "torch._inductor.debug.aot_inductor_minifier_wrapper",
                    side_effect=fake_minifier_wrapper,
                ),
                patch(
                    "torch._inductor.package.package_aoti",
                    side_effect=fake_package_aoti,
                ),
            ):
                result = torch._inductor.aoti_compile_and_package(
                    exported_program,
                    package_path=package_path,
                    inductor_configs=options,
                )

        self.assertEqual(result, package_path)
        self.assertEqual(len(compile_aot_core_calls), 1)
        self.assertEqual(package_aoti_calls, [(package_path, [so_path])])
        self.assertIs(compile_aot_core_calls[0][3], options)
        self.assertTrue(options["aot_inductor.package"])


if __name__ == "__main__":
    run_tests()
