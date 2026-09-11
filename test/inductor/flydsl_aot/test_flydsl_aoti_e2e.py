# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from torch._higher_order_ops.flydsl_kernel_wrap import flydsl_kernel_wrapper_mutation
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._inductor.utils import fresh_cache, run_and_get_cpp_code
from torch.autograd import DeviceType
from torch.testing._internal.common_utils import TestCase


HAS_FLYDSL = torch.cuda.is_available() and runtime_available()
if HAS_FLYDSL:
    from caffe2.test.inductor.flydsl_aot.flydsl_aoti_test_models import (
        BoundReluModel,
        ComposedModel,
        DynamicRMSNormModel,
        RELU_SCALE,
        TwoStageAddModel,
        VectorAddModel,
    )
    from caffe2.test.inductor.flydsl_aot.flydsl_test_kernels import (
        GEMM_K,
        GEMM_M,
        GEMM_N,
        RMS_EPS,
        RMS_N,
    )

RELOCATED_PACKAGE_LOADER_SCRIPT = """
import sys

import torch

package_path = sys.argv[1]
lhs = torch.arange(1024, device="cuda", dtype=torch.float32)
rhs = torch.arange(1024, device="cuda", dtype=torch.float32).flip(0)
loader = torch._C._aoti.AOTIModelPackageLoader(
    package_path,
    "model",
    False,
    1,
    -1,
)
actual = loader.run([lhs, rhs])[0]
torch.cuda.synchronize()
torch.testing.assert_close(actual, lhs + rhs)
"""


@unittest.skipUnless(HAS_FLYDSL, "FlyDSL is not available")
class FlyDSLAOTIEndToEndTest(TestCase):
    def _assert_package_runs_after_relocation(
        self,
        package_path: str,
        root: Path,
    ) -> None:
        relocated_dir = root / "relocated"
        relocated_dir.mkdir()
        relocated_package = relocated_dir / "renamed_model.pt2"
        shutil.move(package_path, relocated_package)
        Path(package_path).parent.rmdir()

        child_cache = root / "child_cache"
        child_cache.mkdir()
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(child_cache)
        env["TRITON_CACHE_DIR"] = str(child_cache / "triton")
        env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                RELOCATED_PACKAGE_LOADER_SCRIPT,
                str(relocated_package),
            ],
            cwd=relocated_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(
            0,
            result.returncode,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_vector_add_runs_from_package(self):
        lhs = torch.randn(1024, device="cuda", dtype=torch.float32)
        rhs = torch.randn_like(lhs)
        exported = torch.export.export(VectorAddModel(), (lhs, rhs), strict=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            producer_dir = root / "producer"
            producer_dir.mkdir()
            with fresh_cache(dir=tmpdir):
                producer_cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
                package_path, generated_code = run_and_get_cpp_code(
                    torch._inductor.aoti_compile_and_package,
                    exported,
                    package_path=str(producer_dir / "flydsl_vector_add.pt2"),
                    inductor_configs={"compile_threads": 1},
                )
            self.assertFalse(producer_cache.exists())
            self.assertNotIn("aoti_torch_clone", generated_code)
            self.assertNotIn("aoti_torch_copy_", generated_code)
            self.assertNotIn("triton_poi", generated_code)
            with zipfile.ZipFile(package_path) as package:
                packaged_files = package.namelist()
            self.assertTrue(
                any(path.endswith("/libfly_jit_runtime.so") for path in packaged_files),
                packaged_files,
            )
            self.assertTrue(
                any(
                    Path(path).name.startswith("libmlir_c_runner_utils.so")
                    for path in packaged_files
                ),
                packaged_files,
            )
            loader = torch._C._aoti.AOTIModelPackageLoader(
                package_path,
                "model",
                False,
                1,
                -1,
            )
            actual = loader.run([lhs, rhs])[0]
            torch.testing.assert_close(actual, lhs + rhs)

            stream = torch.cuda.Stream()
            self.assertNotEqual(
                stream.cuda_stream,
                torch.cuda.default_stream().cuda_stream,
            )
            marker = torch.empty_like(lhs)
            with torch.cuda.stream(stream):
                torch.add(lhs, rhs, out=marker)
                loader.run([lhs, rhs])
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                with torch.cuda.stream(stream):
                    torch.add(lhs, rhs, out=marker)
                    stream_actual = loader.run([lhs, rhs])[0]
                torch.cuda.synchronize()
            gpu_events = [
                event for event in prof.events() if event.device_type == DeviceType.CUDA
            ]
            self.assertGreaterEqual(len(gpu_events), 2)
            self.assertEqual(
                1,
                len({event.device_resource_id for event in gpu_events}),
                [(event.name, event.device_resource_id) for event in gpu_events],
            )
            torch.testing.assert_close(stream_actual, lhs + rhs)

            del loader
            self._assert_package_runs_after_relocation(package_path, root)

    def test_multi_kernel_launcher_runs_from_package(self):
        lhs = torch.randn(1024, device="cuda", dtype=torch.float32)
        rhs = torch.randn_like(lhs)
        exported = torch.export.export(TwoStageAddModel(), (lhs, rhs), strict=True)
        self.assertEqual(
            1,
            len(
                exported.graph_module.graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_mutation,
                )
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with fresh_cache(dir=tmpdir):
                package_path = torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(Path(tmpdir) / "flydsl_two_stage_add.pt2"),
                    inductor_configs={"compile_threads": 1},
                )
                compiled = torch._inductor.aoti_load_package(package_path)
                actual = compiled(lhs, rhs)

        torch.testing.assert_close(actual, torch.sin(lhs + rhs + rhs))

    def test_bound_launcher_runs_eager_and_from_package(self):
        inp = torch.randn(1024, device="cuda", dtype=torch.float32)
        model = BoundReluModel()
        expected = torch.relu(inp) * RELU_SCALE

        eager_actual = model(inp)
        torch.cuda.synchronize()
        torch.testing.assert_close(eager_actual, expected)

        exported = torch.export.export(model, (inp,), strict=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            with fresh_cache(dir=tmpdir):
                package_path = torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(Path(tmpdir) / "flydsl_bound_relu.pt2"),
                    inductor_configs={"compile_threads": 1},
                )
                compiled = torch._inductor.aoti_load_package(package_path)
                aoti_actual = compiled(inp)

        torch.testing.assert_close(aoti_actual, expected)

    def test_dynamic_rms_norm_runs_multiple_row_counts(self):
        inp = torch.randn(4, RMS_N, device="cuda", dtype=torch.float32)
        weight = torch.randn(RMS_N, device="cuda", dtype=torch.float32)
        rows = torch.export.Dim("rows", min=1, max=32)
        exported = torch.export.export(
            DynamicRMSNormModel(),
            (inp, weight),
            dynamic_shapes=({0: rows}, None),
            strict=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with fresh_cache(dir=tmpdir):
                package_path = torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(Path(tmpdir) / "flydsl_dynamic_rms_norm.pt2"),
                    inductor_configs={"compile_threads": 1},
                )
                loader = torch._C._aoti.AOTIModelPackageLoader(
                    package_path,
                    "model",
                    False,
                    1,
                    -1,
                )
                for row_count in (1, 4, 17, 32):
                    test_inp = torch.randn(
                        row_count,
                        RMS_N,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    actual = loader.run([test_inp, weight])[0]
                    expected = (
                        test_inp
                        * torch.rsqrt(
                            test_inp.square().mean(dim=-1, keepdim=True) + RMS_EPS
                        )
                        * weight
                    )
                    torch.testing.assert_close(
                        actual,
                        expected,
                        atol=2e-4,
                        rtol=2e-4,
                    )

    def test_non_contiguous_rms_norm_runs_from_package(self):
        rows = 4
        inp = (
            torch.arange(
                rows * RMS_N,
                device="cuda",
                dtype=torch.float32,
            )
            .reshape(RMS_N, rows)
            .transpose(0, 1)
        )
        weight = torch.randn(RMS_N, device="cuda", dtype=torch.float32)
        self.assertFalse(inp.is_contiguous())
        self.assertEqual((1, rows), inp.stride())
        exported = torch.export.export(
            DynamicRMSNormModel(),
            (inp, weight),
            strict=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with fresh_cache(dir=tmpdir):
                package_path = torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(
                        Path(tmpdir) / "flydsl_non_contiguous_rms_norm.pt2"
                    ),
                    inductor_configs={"compile_threads": 1},
                )
                compiled = torch._inductor.aoti_load_package(package_path)
                runtime_inp = torch.randn(
                    RMS_N,
                    rows,
                    device="cuda",
                    dtype=torch.float32,
                ).transpose(0, 1)
                self.assertEqual(inp.stride(), runtime_inp.stride())
                actual = compiled(runtime_inp, weight)

        expected = (
            runtime_inp
            * torch.rsqrt(runtime_inp.square().mean(dim=-1, keepdim=True) + RMS_EPS)
            * weight
        )
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

    def test_composes_multiple_flydsl_and_pytorch_ops(self):
        torch.manual_seed(0)
        lhs = torch.randn(GEMM_M, GEMM_K, device="cuda", dtype=torch.float32)
        rhs = torch.randn(GEMM_N, GEMM_K, device="cuda", dtype=torch.float32)
        bias = torch.randn(GEMM_N, device="cuda", dtype=torch.float32)
        weight = torch.randn(RMS_N, device="cuda", dtype=torch.float32)
        inputs = (lhs, rhs, bias, weight)
        exported = torch.export.export(ComposedModel(), inputs, strict=True)
        flydsl_nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )
        self.assertEqual(3, len(flydsl_nodes))

        with tempfile.TemporaryDirectory() as tmpdir:
            with fresh_cache(dir=tmpdir):
                package_path = torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(Path(tmpdir) / "flydsl_composed.pt2"),
                    inductor_configs={"compile_threads": 1},
                )
                with zipfile.ZipFile(package_path) as package:
                    packaged_files = package.namelist()
                self.assertEqual(
                    1,
                    sum(
                        path.endswith("/libfly_jit_runtime.so")
                        for path in packaged_files
                    ),
                )
                compiled = torch._inductor.aoti_load_package(package_path)
                actual = compiled(*inputs)

        expected_gemm = lhs @ rhs.T
        expected_activation = torch.relu(expected_gemm + bias) * RELU_SCALE
        mixed = expected_activation + torch.sin(expected_activation) * 0.25
        expected_normalized = (
            mixed
            * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + RMS_EPS)
            * weight
        )
        expected = (
            expected_normalized * 1.5 - 0.5,
            expected_gemm,
            expected_activation,
            expected_normalized,
        )
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                atol=2e-4,
                rtol=2e-4,
            )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
