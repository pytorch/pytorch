# Owner(s): ["module: inductor"]

import json
import tempfile
import unittest
import uuid
from io import StringIO
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch._inductor.analysis.profile_analysis import (
    _augment_trace_helper,
    _create_extern_mapping,
    main,
)
from torch._inductor.utils import fresh_inductor_cache, tabulate_2d, zip_dicts
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    skipIf,
)
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfXpu,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.inductor_utils import IS_BIG_GPU


example_profile = """
{
  "schemaVersion": 1,
  "deviceProperties": [
    {
      "id": 0, "name": "NVIDIA H100", "totalGlobalMem": 101997215744,
      "computeMajor": 9, "computeMinor": 0,
      "maxThreadsPerBlock": 1024, "maxThreadsPerMultiprocessor": 2048,
      "regsPerBlock": 65536, "warpSize": 32,
      "sharedMemPerBlock": 49152, "numSms": 132
    , "regsPerMultiprocessor": 65536, "sharedMemPerBlockOptin": 232448, "sharedMemPerMultiprocessor": 233472
    }
  ],
  "cupti_version": 24,
  "cuda_runtime_version": 12060,
  "with_flops": 1,
  "record_shapes": 1,
  "cuda_driver_version": 12040,
  "profile_memory": 1,
  "trace_id": "301995E163ED42048FBD783860E6E7DC",
  "displayTimeUnit": "ms",
  "baseTimeNanoseconds": 1743521598000000000,
  "traceEvents": [
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::convolution", "pid": 1147039, "tid": 1147039,
    "ts": 198093488368.463, "dur": 425.453,
    "args": {
      "External id": 1340,"Sequence number": 0, "Fwd thread id": 0, "Record function id": 0, "Concrete Inputs": \
["", "", "", "[2, 2]", "[3, 3]", "[1, 1]", "False", "[0, 0]", "1"], "Input type": ["float", "float", "", \
"ScalarList", "ScalarList", "ScalarList", "Scalar", "ScalarList", "Scalar"], "Input Strides": [[150528, 1, 672, 3],\
[147, 1, 21, 3], [], [], [], [], [], [], []], "Input Dims": [[1, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], \
[], []], "Ev Idx": 1339
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::_convolution", "pid": 1147039, "tid": 1147039,
    "ts": 198093488444.498, "dur": 341.867,
    "args": {
      "External id": 1341,"Record function id": 0, "Concrete Inputs": ["", "", "", "[2, 2]", "[3, 3]", "[1, 1]",\
 "False", "[0, 0]", "1", "False", "False", "True", "True"], "Input type": ["float", "float", "", "ScalarList",\
 "ScalarList", "ScalarList", "Scalar", "ScalarList", "Scalar", "Scalar", "Scalar", "Scalar", "Scalar"], "Input Strides": \
[[150528, 1, 672, 3], [147, 1, 21, 3], [], [], [], [], [], [], [], [], [], [], []], "Input Dims": [[1, 3, 224, 224], \
[64, 3, 7, 7], [], [], [], [], [], [], [], [], [], [], []], "Ev Idx": 1340
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::addmm", "pid": 1147039, "tid": 1147039,
    "ts": 198093513655.849, "dur": 251.130,
    "args": {
      "External id": 1619,"Sequence number": 0, "Fwd thread id": 0, "Record function id": 0, "Concrete Inputs": \
["", "", "", "1", "1", ""], "Input type": ["float", "float", "float", "Scalar", "Scalar", "float"], "Input Strides":\
 [[1], [0, 1], [1, 2048], [], [], [1000, 1]], "Input Dims": [[1000], [1, 2048], [2048, 1000], [], [], [1, 1000]], \
"Ev Idx": 1618
    }
  },
  {
    "ph": "X", "cat": "kernel", "name": "void cutlass_addmm", "pid": 1147039, "tid": 1147039,
    "ts": 198093513655.849, "dur": 251.130,
    "args": {
      "External id": 1619,"Sequence number": 0, "Fwd thread id": 0, "Record function id": 0,  "Ev Idx": 1618
    }
  },
  {
    "ph": "X", "cat": "kernel", "name": "void convolution_kernel", "pid": 1147039, "tid": 1147039,
    "ts": 198093513655.849, "dur": 200.130,
    "args": {
      "External id": 1342, "Sequence number": 0, "Fwd thread id": 0, "Record function id": 0,  "Ev Idx": 1618
    }
  },
  {
    "ph": "X", "cat": "cpu_op", "name": "aten::convolution", "pid": 1147039, "tid": 1147039,
    "ts": 198093488444.498, "dur": 341.867,
    "args": {
      "External id": 1342,"Record function id": 0, "Concrete Inputs": ["", "", "", "[2, 2]", "[3, 3]", "[1, 1]", \
"False", "[0, 0]", "1", "False", "False", "True", "True"], "Input type": ["float", "float", "", "ScalarList", \
"ScalarList", "ScalarList", "Scalar", "ScalarList", "Scalar", "Scalar", "Scalar", "Scalar", "Scalar"], "Input \
Strides": [[150528, 1, 672, 3], [147, 1, 21, 3], [], [], [], [], [], [], [], [], [], [], []], "Input Dims": \
[[1, 3, 224, 224], [64, 3, 7, 7], [], [], [], [], [], [], [], [], [], [], []], "Ev Idx": 1340
    }
  }
],
  "traceName": "/tmp/compiled_module_profile.json"
}
"""


def verify_flops(self, expected_flops, out_profile):
    j = 0
    for i in range(len(out_profile["traceEvents"])):
        if "kernel_flop" in out_profile["traceEvents"][i]["args"]:
            self.assertEqual(
                out_profile["traceEvents"][i]["args"]["kernel_flop"],
                expected_flops[j],
            )
            j += 1


def random_tensor(size, dtype, **kwargs):
    if dtype in [torch.half, torch.bfloat16, torch.float, torch.double]:
        return torch.randn(size, dtype=dtype, **kwargs)
    elif dtype in [torch.uint8, torch.int8, torch.short, torch.int, torch.long]:
        return torch.randint(0, 100, size, dtype=dtype, **kwargs)
    else:
        raise ValueError("Unsupported data type")


def cT(device, dtype):
    def T(*shape, requires_grad=False):
        return random_tensor(
            shape, requires_grad=requires_grad, device=device, dtype=dtype
        )

    return T


def FlopCounterMode(*args, **kwargs):
    return torch.utils.flop_counter.FlopCounterMode(*args, **kwargs, display=False)


TMP_DIR = tempfile.mkdtemp()


def trace_files():
    TRACE1 = f"{TMP_DIR}/trace1-{uuid.uuid4()}.json"
    TRACE2 = f"{TMP_DIR}/trace2-{uuid.uuid4()}.json"
    return TRACE1, TRACE2


def _test_model(device, dtype, compile=True, addmm=True, bmm=True):
    T = cT(device, dtype)

    def model():
        input_conv = T(1, 3, 56, 56)
        conv_weight = T(12, 3, 5, 5)

        # Increased matrix sizes
        B = 8
        M = 256
        N = 512
        K = 768
        mat1 = T(M, N)
        mat2 = T(N, K)

        batch_mat1 = T(B, M, N)
        batch_mat2 = T(B, N, K)

        conv_output = F.conv2d(input_conv, conv_weight)

        conv_output = conv_output * 10
        mm_output = torch.mm(mat1, mat2)
        ret = [
            conv_output.flatten(),
            mm_output.flatten(),
        ]

        if addmm:
            addmm_output = torch.addmm(
                torch.zeros(mm_output.shape, device=mat1.device, dtype=mat1.dtype),
                mat1,
                mat2,
            )
            ret.append(addmm_output.flatten())

        if bmm:
            bmm_output = torch.bmm(batch_mat1, batch_mat2)
            ret.append(bmm_output.flatten())

        if bmm and addmm:
            baddbmm_output = torch.baddbmm(
                torch.zeros(
                    1,
                    *mm_output.shape,
                    device=batch_mat1.device,
                    dtype=batch_mat1.dtype,
                ),
                batch_mat1,
                batch_mat2,
            )
            ret.append(baddbmm_output.flatten())

        return torch.cat(ret)

    if compile:
        return torch.compile(
            model, options={"benchmark_kernel": True, "profile_bandwidth": True}
        )
    return model


def _pointwise_test_model(device, dtype, compile=True):
    T = cT(device, dtype)

    def model():
        M = 1024
        N = 512
        mat3 = T(M, N)
        mat4 = T(M, N)
        pointwise_output = torch.add(mat3, mat4).sin()
        return pointwise_output

    if compile:
        return torch.compile(
            model, options={"benchmark_kernel": True, "profile_bandwidth": True}
        )
    return model


prefix = ["profile.py"]


class TestUtils(TestCase):
    def test_tabulate2d(self):
        headers = ["Kernel", "Self H100 TIME (ms)", "Count", "Percent"]
        rows = [
            ["aten::mm", 0.500, 7, 0.0],
            ["aten::bmm", 0.400, 6, 0.0],
            ["aten::baddbmm", 0.300, 5, 0.0],
            ["aten::convolution", 0.200, 4, 0.0],
            ["aten::cudnn_convolution", 0.100, 3, 0.0],
        ]
        table = [
            " Kernel                  | Self H100 TIME (ms) | Count | Percent ",
            "-----------------------------------------------------------------",
            " aten::mm                |                 0.5 |     7 |     0.0 ",
            " aten::bmm               |                 0.4 |     6 |     0.0 ",
            " aten::baddbmm           |                 0.3 |     5 |     0.0 ",
            " aten::convolution       |                 0.2 |     4 |     0.0 ",
            " aten::cudnn_convolution |                 0.1 |     3 |     0.0 ",
        ]
        res = tabulate_2d(rows, headers)
        for r, t in zip(res.split("\n"), table):
            self.assertEqual(r, t)

    def test_zip_dicts(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"a": 3, "c": 4}
        res1 = zip_dicts(d1, d2, d1_default=32, d2_default=48)
        self.assertEqual(set(res1), {("a", 1, 3), ("b", 2, 48), ("c", 32, 4)})
        res2 = zip_dicts(d1, d2)
        self.assertEqual(set(res2), {("a", 1, 3), ("b", 2, None), ("c", None, 4)})


def has_supported_gpu():
    """Check if any GPU platform with Triton support is available."""
    return torch.xpu.is_available() or SM80OrLater or torch.version.hip


class TestAnalysis(TestCase):
    @skipIf(not has_supported_gpu(), "Requires XPU, CUDA SM80+, or ROCm")
    def test_noop(self):
        with (
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
            patch("sys.argv", [*prefix]),
        ):
            main()
            self.assertEqual(mock_stdout.getvalue(), "")

    @skipIf(not has_supported_gpu(), "Requires XPU, CUDA SM80+, or ROCm")
    @dtypes(torch.float, torch.double, torch.float16)
    def test_diff(self, device, dtype):
        """
        diff, testing out the nruns feature too.
        """
        if device == "cpu" or torch.version.hip is not None:
            # TODO cpu support
            return
        om = _test_model(device, dtype)
        REPEAT = 5
        trace1, trace2 = trace_files()
        print(f"first trace {trace1}")
        torch._dynamo.reset()  # reset the cache
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p:
                om()
        p.export_chrome_trace(trace1)

        print(f"second trace {trace2}")
        torch._dynamo.reset()  # reset the cache
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p:
                for _ in range(REPEAT):
                    om()
        p.export_chrome_trace(trace2)

        print("diffing...")
        with patch(
            "sys.argv",
            [
                *prefix,
                "--diff",
                trace1,
                "foo",
                trace2,
                "bar",
                str(dtype).split(".")[-1],
                "--name_limit",
                "30",
            ],
        ):
            main()

    @skipIf(not (SM80OrLater or TEST_XPU), "Requires SM80 or XPU")
    def test_augment_trace_helper_unit(self):
        js = json.loads(example_profile)
        out_profile = _augment_trace_helper(js)
        expected_flops = [4096000, 4096000, 223552896, 223552896, 0, 0, 0]
        verify_flops(self, expected_flops, out_profile)

    @skipIf(not has_supported_gpu(), "Requires XPU, CUDA SM80+, or ROCm")
    @dtypes(torch.float, torch.double, torch.float16)
    @parametrize(
        "maxat",
        [
            (True, "TRITON"),
        ],
    )
    @skipIf(not IS_BIG_GPU, "we can't use Triton only as a backend for max autotune")
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_triton_has_metadata(self, device, dtype, maxat):
        """
        make sure that the chrome trace of triton kernels contains certain values
        """
        if device == "cpu" or torch.version.hip is not None:
            return

        T = cT(device, dtype)
        input_conv = T(1, 3, 56, 56)
        conv_weight = T(12, 3, 5, 5)

        def om(i, w):
            # Convolution operation
            conv_output = F.conv2d(i, w)
            return conv_output

        max_autotune, backends = maxat
        comp_omni = torch.compile(
            om,
            options={
                "benchmark_kernel": True,
                "max_autotune_gemm_backends": backends,
                "max_autotune": max_autotune,
            },
        )

        def verify_triton(comp):
            torch._dynamo.reset()  # reset the cache
            with fresh_inductor_cache():
                with torch.profiler.profile(record_shapes=True) as profile:
                    comp(input_conv, conv_weight)

            trace1, _ = trace_files()
            profile.export_chrome_trace(trace1)
            with open(trace1) as f:
                out_profile = json.load(f)
            seen = False
            for event in out_profile["traceEvents"]:
                if "triton" in event["name"] and "conv" in event["name"]:
                    seen = True
            self.assertTrue(seen, "no triton conv found")

        verify_triton(comp_omni)

    @skipIf(not has_supported_gpu(), "Requires XPU, CUDA SM80+, or ROCm")
    @skipIfXpu(
        msg="Intel triton issue: https://github.com/intel/intel-xpu-backend-for-triton/issues/5491"
    )
    @dtypes(torch.float, torch.float16)
    @parametrize(
        "maxat",
        [
            (False, "ATEN,TRITON"),
            (True, "ATEN,TRITON"),
            (True, "ATEN"),
            (True, "TRITON"),
        ],
    )
    @unittest.skipIf(
        not IS_BIG_GPU, "we can't use Triton only as a backend for max autotune"
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_augment_trace_against_flop_counter(self, device, dtype, maxat):
        # this tests to see if we can only use a Triton backend for max autotune
        max_autotune, backends = maxat
        if device == "cpu" or torch.version.hip is not None:
            return
        om = _test_model(device, dtype, compile=False)

        comp_omni = torch.compile(
            om,
            options={
                "benchmark_kernel": True,
                "max_autotune_gemm_backends": backends,
                "max_autotune": max_autotune,
            },
        )
        comp_omni()

        torch._dynamo.reset()  # reset the cache
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as profile:
                comp_omni()

        torch._dynamo.reset()  # reset the cache
        with fresh_inductor_cache():
            with FlopCounterMode() as mode:
                comp_omni()

        trace1, trace2 = trace_files()
        profile.export_chrome_trace(trace1)
        with patch(
            "sys.argv",
            [*prefix, "--augment_trace", trace1, trace2, str(dtype).split(".")[-1]],
        ):
            main()

        with open(trace2) as f:
            out_profile = json.load(f)

        flop_counts = mode.flop_counts
        extern_mapping = _create_extern_mapping(out_profile)

        seen_mm = False
        seen_bmm = False
        seen_baddbmm = False
        seen_conv = False
        for event in out_profile["traceEvents"]:
            if (
                "cat" not in event
                or event["cat"] != "kernel"
                or "args" not in event
                or "External id" not in event["args"]
            ):
                continue

            external_op = extern_mapping[event["args"]["External id"]][0]
            name: str = external_op["name"]
            self.assertNotEqual(name, None)
            self.assertEqual(type(name), str)
            if name.startswith("aten::mm") or "_mm_" in name:
                seen_mm = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.mm],
                )
            if (
                name.startswith(
                    (
                        "aten::cudnn_convolution",
                        "aten::convolution",
                        "aten::_convolution",
                        "aten::convolution_overrideable",
                    )
                )
                or "conv" in name
            ):
                seen_conv = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.convolution],
                )
            if name.startswith("aten::baddbmm") or "_baddbmm_" in name:
                seen_baddbmm = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.baddbmm],
                )
            if name.startswith("aten::bmm") or "_bmm_" in name:
                seen_bmm = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.bmm],
                )
        self.assertTrue(seen_mm)
        self.assertTrue(seen_bmm)
        self.assertTrue(seen_baddbmm)
        self.assertTrue(seen_conv)

    @skipIf(not has_supported_gpu(), "Requires XPU, CUDA SM80+, or ROCm")
    @dtypes(torch.float, torch.float16)
    @parametrize(
        "maxat",
        [
            (False, "ATEN,TRITON"),
            (True, "ATEN,TRITON"),
            (True, "ATEN"),
            (True, "TRITON"),
        ],
    )
    @unittest.skipIf(
        not IS_BIG_GPU, "we can't use Triton only as a backend for max autotune"
    )
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_pointwise_bandwidth(self, device, dtype, maxat):
        # this tests to see if we can only use a Triton backend for max autotune
        max_autotune, backends = maxat
        if device == "cpu" or torch.version.hip is not None:
            return
        om = _pointwise_test_model(device, dtype, compile=False)
        comp_omni = torch.compile(
            om,
            options={
                "benchmark_kernel": True,
                "max_autotune_gemm_backends": backends,
                "max_autotune": max_autotune,
            },
        )
        comp_omni()

        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as profile:
                comp_omni()
        trace1, _ = trace_files()
        profile.export_chrome_trace(trace1)

        with patch(
            "sys.argv",
            [*prefix, "--analysis", trace1, str(dtype).split(".")[-1]],
        ):
            main()

        with open(trace1) as f:
            out_profile = json.load(f)

        for event in out_profile["traceEvents"]:
            if event["name"] == "triton_poi_fused_add_randn_sin_0":
                event["args"]["kernel_num_gb"] = 0.002097168

    @skipIf(not has_supported_gpu(), "Requires XPU, CUDA SM80+, or ROCm")
    @dtypes(torch.float, torch.float16)
    def test_combine_profiles(self, device, dtype):
        """
        Test combining multiple profiles into a single profile.
        """
        if device == "cpu" or torch.version.hip is not None:
            return

        # Create three different models to generate different traces
        om1 = _test_model(device, dtype, addmm=True, bmm=False)
        om2 = _test_model(device, dtype, addmm=False, bmm=True)
        om3 = _pointwise_test_model(device, dtype)

        # Generate three separate traces
        trace1, trace2 = trace_files()
        trace3 = f"{TMP_DIR}/trace3-{uuid.uuid4()}.json"
        combined_trace = f"{TMP_DIR}/combined-{uuid.uuid4()}.json"

        # Generate first trace
        torch._dynamo.reset()
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p1:
                om1()
        p1.export_chrome_trace(trace1)

        # Generate second trace
        torch._dynamo.reset()
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p2:
                om2()
        p2.export_chrome_trace(trace2)

        # Generate third trace
        torch._dynamo.reset()
        with fresh_inductor_cache():
            with torch.profiler.profile(record_shapes=True) as p3:
                om3()
        p3.export_chrome_trace(trace3)

        # Combine the three traces
        with patch(
            "sys.argv",
            [
                *prefix,
                "--combine",
                trace1,
                trace2,
                trace3,
                combined_trace,
            ],
        ):
            main()

        # Verify the combined trace exists and contains expected data
        with open(combined_trace) as f:
            combined_profile = json.load(f)

        # Load original traces for comparison
        with open(trace1) as f:
            profile1 = json.load(f)
        with open(trace2) as f:
            profile2 = json.load(f)
        with open(trace3) as f:
            profile3 = json.load(f)

        # Verify trace events are combined
        expected_event_count = (
            len(profile1["traceEvents"])
            + len(profile2["traceEvents"])
            + len(profile3["traceEvents"])
        )
        self.assertEqual(len(combined_profile["traceEvents"]), expected_event_count)

        # Verify device properties are present
        self.assertIn("deviceProperties", combined_profile)
        # XPU currently does not have the deviceProperties like CUDA.
        # See https://github.com/intel/torch-xpu-ops/issues/2247
        if torch.cuda.is_available():
            self.assertGreater(len(combined_profile["deviceProperties"]), 0)

        # Verify some trace events from each original profile are present
        combined_event_names = {
            event["name"] for event in combined_profile["traceEvents"]
        }

        # Check that we have events from each original profile
        profile1_event_names = {event["name"] for event in profile1["traceEvents"]}
        profile2_event_names = {event["name"] for event in profile2["traceEvents"]}
        profile3_event_names = {event["name"] for event in profile3["traceEvents"]}

        # At least some events from each profile should be in the combined profile
        self.assertTrue(profile1_event_names.intersection(combined_event_names))
        self.assertTrue(profile2_event_names.intersection(combined_event_names))
        self.assertTrue(profile3_event_names.intersection(combined_event_names))


instantiate_device_type_tests(TestAnalysis, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
