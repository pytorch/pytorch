# Owner(s): ["module: inductor"]

import json
import tempfile
import uuid
from io import StringIO
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch.utils.flop_counter
from torch._inductor.analysis.profile_analysis import (
    _augment_trace_helper,
    _create_extern_mapping,
    JsonProfile,
    main,
)
from torch._inductor.utils import tabulate_2d, zip_dicts
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    skipIf,
)
from torch.testing._internal.common_utils import run_tests, TestCase, parametrize


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


def omni_model(device, dtype):
    T = cT(device, dtype)

    def model():
        input_conv = T(1, 3, 56, 56)
        conv_weight = T(12, 3, 5, 5)

        # Increased matrix sizes
        mat1 = T(400, 600)
        mat2 = T(600, 800)

        batch_mat1 = T(1, 600, 800)
        batch_mat2 = T(1, 800, 480 * 48)

        # Convolution operation
        conv_output = F.conv2d(input_conv, conv_weight)

        # a pointwise op
        conv_output = conv_output * 10

        # Matrix multiplication (addmm) operation
        addmm_output = torch.addmm(
            torch.zeros(400, 800, device=mat1.device, dtype=mat1.dtype), mat1, mat2
        )

        # Batch matrix multiplication (bmm) operation
        bmm_output = torch.bmm(batch_mat1, batch_mat2)

        # Batch addition matrix multiplication (baddbmm) operation
        baddbmm_output = torch.baddbmm(
            torch.zeros(
                1, 600, 23040, device=batch_mat1.device, dtype=batch_mat1.dtype
            ),
            batch_mat1,
            batch_mat2,
        )

        mm_output = torch.mm(mat1, mat2)

        return torch.cat(
            [
                conv_output.flatten(),
                addmm_output.flatten(),
                bmm_output.flatten(),
                baddbmm_output.flatten(),
                mm_output.flatten(),
            ]
        )

    return torch.compile(
        model, options={"benchmark_kernel": True, "profile_bandwidth": True}
    )


prefix = ["profile.py"]


class TestUtils(TestCase):
    def test_tabulate2d(self):
        headers = ["Kernel", "Self H100 TIME (ms)", "Count", "Percent"]
        rows = [
            ["aten::mm", 0.000, 1, 0.0],
            ["aten::bmm", 0.000, 1, 0.0],
            ["aten::baddbmm", 0.000, 1, 0.0],
            ["aten::convolution", 0.000, 1, 0.0],
            ["aten::cudnn_convolution", 0.000, 1, 0.0],
        ]
        table = [
            " Kernel                  | Self H100 TIME (ms) | Count | Percent ",
            "-----------------------------------------------------------------",
            " aten::mm                |                 0.0 |     1 |     0.0 ",
            " aten::bmm               |                 0.0 |     1 |     0.0 ",
            " aten::baddbmm           |                 0.0 |     1 |     0.0 ",
            " aten::convolution       |                 0.0 |     1 |     0.0 ",
            " aten::cudnn_convolution |                 0.0 |     1 |     0.0 ",
        ]
        res = tabulate_2d(rows, headers)
        for r, t in zip(res.split("\n"), table):
            self.assertEqual(r, t)

    def test_zip_dicts(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"a": 3, "c": 4}
        res = zip_dicts(d1, d2, d1_default="foo", d2_default="bar")
        self.assertEqual(set(res), {("a", 1, 3), ("b", 2, "bar"), ("c", "foo", 4)})
        res = zip_dicts(d1, d2)
        self.assertEqual(set(res), {("a", 1, 3), ("b", 2, None), ("c", None, 4)})


class TestAnalysis(TestCase):
    @skipIf(not SM70OrLater, "Requires sm70")
    def test_noop(self):
        with (
            patch("sys.stdout", new_callable=StringIO) as mock_stdout,
            patch("sys.argv", [*prefix]),
        ):
            main()
            self.assertEqual(mock_stdout.getvalue(), "")

    @skipIf(not SM70OrLater, "Requires sm70")
    @dtypes(torch.float, torch.double)
    def test_diff(self, device, dtype):
        """
        diff, testing out the nruns feature too.
        """
        om = omni_model(device, dtype)
        REPEAT = 5
        trace1, trace2 = trace_files()
        with torch.profiler.profile(record_shapes=True) as p:
            om()
        p.export_chrome_trace(trace1)

        with torch.profiler.profile(record_shapes=True) as p:
            for _ in range(REPEAT):
                om()
        p.export_chrome_trace(trace2)

        with patch(
            "sys.argv",
            [
                *prefix,
                "--diff",
                trace1,
                "1",
                "foo",
                trace2,
                str(REPEAT),
                "bar",
                "--name_limit",
                "200",
            ],
        ):
            main()

    @skipIf(not SM70OrLater, "Requires sm70")
    def test_augment_trace_helper(self):
        js = json.loads(example_profile)
        out_profile = _augment_trace_helper(js)
        expected_flops = [4096000, 4096000, 223552896, 223552896, 0, 0, 0]
        verify_flops(self, expected_flops, out_profile)

    @skipIf(not SM70OrLater, "Requires sm70")
    @dtypes(torch.float, torch.double)
    def test_augment_trace_helper_args(self, device, dtype):
        om = omni_model(device, dtype)
        with torch.profiler.profile(record_shapes=True) as p:
            om()
        trace1, trace2 = trace_files()
        p.export_chrome_trace(trace1)
        with patch("sys.argv", [*prefix, "--augment_trace", trace1, trace2]):
            main()
        profile = JsonProfile(trace2, 1, "foo")
        rep = profile.report()
        # If these fail, just update them. They could change over time
        if device != "cpu":
            self.assertTrue(len(rep.split("\n")) > 4)
        self.assertIn("Kernel Name", rep)
        self.assertIn("Kernel Count", rep)
        self.assertIn("FLOPS", rep)
        self.assertIn("bw gbps", rep)
        self.assertIn("Dur (ms)", rep)
        self.assertIn("Achieved", rep)
        self.assertIn("|", rep)
        self.assertIn("-----", rep)

        tables = profile._create_tables(profile._devices)
        # check to make sure none of the cols are all zero, no empty columns
        for tab in tables.values():
            header, rows = tab
            ncols = len(header) - 1
            seen = [False] * ncols
            for row in rows.values():
                for i in range(len(row)):
                    try:
                        val = float(row[i])
                    except Exception:
                        continue
                    seen[i] = seen[i] or (val != 0.0)

            if device != "cpu":
                for i in range(len(seen)):
                    self.assertTrue(
                        seen[i],
                        f"column values from column {i + 1} with header '{header[i + 1]}' are all zero",
                    )

        # check to make sure all % values are less than 100%
        percents = []
        for tab in tables.values():
            header, rows = tab
            for i, h in enumerate(header):
                if "%" in h:
                    percents.append(i)
            self.assertTrue(len(percents) > 0, "There are no headers with % in them")
            for row in rows.values():
                for p in percents:
                    idx = p - 1
                    self.assertTrue(
                        float(row[idx]) <= 100.0,
                        f"column values from column {idx} with header '{header[idx]}' is greater than 100%: {row[idx]}",
                    )
                    self.assertTrue(
                        float(row[idx]) >= 0.0,
                        f"column values from column {idx} with header '{header[idx]}' is less than 0%: {row[idx]}",
                    )

    @skipIf(not SM70OrLater, "Requires sm70")
    @dtypes(torch.float, torch.double)
    @parametrize("backends", ["ATEN,TRITON", "TRITON"])
    def test_augment_trace_against_flop_counter(self, device, dtype, backends):
        if device == "cpu":
            return
        om = omni_model(device, dtype)
        comp_omni = torch.compile(
            om, options={"benchmark_kernel": True, "profile_bandwidth": True, "max_autotune_gemm_backends": backends, "force_disable_caches": True, "max_autotune": True}
        )
        comp_omni()

        with torch.profiler.profile(record_shapes=True) as p:
            comp_omni()

        with FlopCounterMode() as mode:
            comp_omni()

        trace1, trace2 = trace_files()
        p.export_chrome_trace(trace1)
        with patch("sys.argv", [*prefix, "--augment_trace", trace1, trace2]):
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
            if "cat" not in event or event["cat"] != "kernel":
                continue

            external_op = extern_mapping[event["args"]["External id"]][0]
            if external_op["name"].startswith("aten::mm"):
                seen_mm = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.mm],
                )
            if (
                external_op["name"].startswith("aten::cudnn_convolution")
                or external_op["name"].startswith("aten::convolution")
                or external_op["name"].startswith("aten::_convolution")
            ):
                seen_conv = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.convolution],
                )
            if external_op["name"].startswith("aten::baddbmm"):
                seen_baddbmm = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.baddbmm],
                )
            if external_op["name"].startswith("aten::bmm"):
                seen_bmm = True
                self.assertEqual(
                    event["args"]["kernel_flop"],
                    flop_counts["Global"][torch.ops.aten.bmm],
                )
        breakpoint()
        self.assertTrue(seen_mm)
        self.assertTrue(seen_bmm)
        self.assertTrue(seen_baddbmm)
        self.assertTrue(seen_conv)


instantiate_device_type_tests(TestAnalysis, globals())

if __name__ == "__main__":
    run_tests()
