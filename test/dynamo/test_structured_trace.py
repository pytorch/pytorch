# Owner(s): ["module: dynamo"]
import copy
import functools
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging.structured
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.test_case import TestCase
from torch._logging._internal import TorchLogsFormatter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port
from torch.testing._internal.inductor_utils import HAS_CUDA


HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")
requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def example_training_fn(a):
    output = a.mul(torch.ones(1000, 1000, requires_grad=True))
    output = output.add(torch.ones(1000, 1000))
    output.sum().backward()
    return output


def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    output = a.add(torch.ones(1000, 1000, device="cuda"))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def replace_dynamic(buffer, key):
    return re.sub(r'("' + key + r'":\s*)(\d+\.\d+)', r"\1<dynamic>", buffer)


class StructuredTraceTestingFilter(logging.Filter):
    def __init__(self, match_name=None):
        self.match_name = match_name

    def filter(self, record):
        if "str" in record.metadata:
            return False
        if self.match_name is not None:
            if "artifact" in record.metadata:
                if self.match_name != record.metadata["artifact"]["name"]:
                    return False
            elif self.match_name not in record.metadata:
                return False
        return True


class ChromiumEventFilter(logging.Filter):
    def filter(self, record):
        return "chromium_event" not in record.metadata


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


class StructuredTraceTestingFormatter(logging.Formatter):
    def format(self, record):
        metadata = copy.deepcopy(record.metadata)

        # Stub out values that are not stable across runs
        # TODO: Check that these match schema
        if "has_payload" in metadata:
            metadata["has_payload"] = "HASH"
        if "dynamo_start" in metadata:
            metadata["dynamo_start"]["stack"] = "STACK"
        if "inductor_output_code" in metadata:
            metadata["inductor_output_code"]["filename"] = "FILENAME"
        if "stack" in metadata:
            metadata["stack"] = "STACK"
        if "compilation_metrics" in metadata:
            metadata["compilation_metrics"] = "METRICS"
        if "bwd_compilation_metrics" in metadata:
            metadata["bwd_compilation_metrics"] = "METRICS"
        if "describe_storage" in metadata:
            metadata["describe_storage"]["describer_id"] = "ID"
        if "describe_tensor" in metadata:
            metadata["describe_tensor"]["describer_id"] = "ID"
            if "view_func" in metadata["describe_tensor"]:
                metadata["describe_tensor"]["view_func"] = "VIEW_FUNC"
        if "describe_source" in metadata:
            metadata["describe_source"]["describer_id"] = "ID"
        if (
            (k := "create_symbol") in metadata
            or (k := "guard_added_fast") in metadata
            or (k := "create_unbacked_symbol") in metadata
        ):
            metadata[k]["user_stack"] = "STACK"
            metadata[k]["stack"] = "STACK"

        if "dump_file" in metadata:
            # Don't include the actually key number, that's sensitive to other
            # test runs
            metadata["dump_file"]["name"] = "<eval_with_key>"
            return (
                json.dumps(metadata)
                + "\n"
                + "\n".join(l.rstrip() for l in record.payload.splitlines())
            )

        return json.dumps(metadata)


trace_log = logging.getLogger("torch.__trace")

chrome_event_filter = ChromiumEventFilter()


def show_chrome_events(fn):
    """
    Don't hide chrome events for this test
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.handler.removeFilter(chrome_event_filter)
        return fn(self, *args, **kwargs)

    return wrapper


class StructuredTraceTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.buffer = io.StringIO()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTraceTestingFormatter())
        self.handler.addFilter(StructuredTraceTestingFilter())
        self.handler.addFilter(chrome_event_filter)
        trace_log.addHandler(self.handler)

        self.raw_file = tempfile.NamedTemporaryFile(
            mode="w", delete=True
        )  # set this to False to keep temporary files
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        trace_log.setLevel(self.old_level)

    def assertParses(self):
        out = tempfile.mkdtemp()
        try:
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    @requires_cuda
    def test_schedule(self):
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_cuda
    def test_cudagraphs(self):
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_recompiles(self):
        def fn(x, y):
            return torch.add(x, y)

        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['y']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_y_": [1000, 1000], "l_x_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s0", "val": "1", "vr": "[-int_oo, int_oo]", "source": "L['y']", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1000, 1000], "add": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_example_fn(self):
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000], "ones_1": [1000, 1000], "output_1": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_example_training_fn(self):
        fn_opt = torch.compile(example_training_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, requires_grad=True))
        buffer = self.buffer.getvalue()
        buffer = replace_dynamic(buffer, "inductor_compile_time_s")
        buffer = replace_dynamic(buffer, "code_gen_time_s")
        buffer = replace_dynamic(buffer, "structured_logging_overhead_s")
        self.assertExpectedInline(
            buffer,
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack1']"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_cpp_guards_str": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack0']"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['___stack0']"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_output_graph": {"sizes": {"l_stack0_": [1000, 1000], "ones": [1000, 1000], "output": [1000, 1000], "sum_1": []}}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_joint_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"aot_backward_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"bwd_compilation_metrics": "METRICS", "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['output']"}, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_dynamo_error(self):
        try:
            fn_opt = torch.compile(dynamo_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_error", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_inductor_error(self):
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError

        # inject an error in the lowerings
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        with unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries):
            try:
                fn_opt = torch.compile(inductor_error_fn, backend="inductor")
                fn_opt(*ARGS)
            except Exception:
                pass

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4000000}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1000, 1000], "is_leaf": true, "requires_grad": true, "stride": [1000, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1000, 1000], "output": [1000, 1000]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "dynamo_error", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_distributed()
    @requires_cuda
    def test_ddp_graphs(self):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                return self.layers(x)

        # TODO: this isn't safely bracketed, will leak
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        model = DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        ddp_model = torch.compile(model, backend="inductor")

        ddp_model(torch.randn(1024, 1024, device="cuda:0"))

        dist.destroy_process_group()

        if not torch._dynamo.config.inline_inbuilt_nn_modules:
            self.assertExpectedInline(
                self.buffer.getvalue(),
                """\
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1024, 1024], "l__self___layers_0": [1024, 1024], "l__self___layers_1": [1024, 1024]}}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_0"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_1"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_hash", "encoding": "json"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_hash", "encoding": "json"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
            )
        else:
            self.assertExpectedInline(
                self.buffer.getvalue(),
                """\
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 1, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 2, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 3, "frame_compile_id": 0, "attempt": 0}
{"dynamo_start": {"stack": "STACK"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 2, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 2, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 2, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 2, "source": "L['x']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 3, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 8, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 3, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 8, "source": "L['self']._modules['layers']._modules['1']._parameters['weight']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 4, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 9, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 4, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 9, "source": "L['self']._modules['layers']._modules['1']._parameters['bias']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_self_modules_layers_modules_0_parameters_weight_": [1024, 1024], "l_self_modules_layers_modules_0_parameters_bias_": [1024], "l_x_": [1024, 1024], "l_self_modules_layers_modules_1_parameters_weight_": [1024, 1024], "l_self_modules_layers_modules_1_parameters_bias_": [1024], "input_1": [1024, 1024], "input_2": [1024, 1024]}}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_0"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"optimize_ddp_split_child": {"name": "submod_1"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "stride": [1024, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['self']._modules['layers']._modules['0']._parameters['weight']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 2, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 2, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 2, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 2, "source": "L['self']._modules['layers']._modules['0']._parameters['bias']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"describe_storage": {"id": 16, "describer_id": "ID", "size": 4194304}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 29, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024, 1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1024, 1], "storage": 16, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 29, "source": "L['self']._modules['layers']._modules['1']._parameters['weight']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 17, "describer_id": "ID", "size": 4096}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 30, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cuda', index=0)", "size": [1024], "is_leaf": true, "requires_grad": true, "is_parameter": true, "stride": [1], "storage": 17, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 30, "source": "L['self']._modules['layers']._modules['1']._parameters['bias']"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
{"aot_joint_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_forward_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_backward_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "rank": 0, "frame_id": 4, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
            )

        self.assertParses()

    @requires_tlparse
    def test_graph_breaks(self):
        @torch.compile(backend="inductor")
        def fn(x):
            torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(1))

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "dynamo_graph_break_reason", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 1, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 1}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "add": [1]}}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 1, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 1, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    # TODO: bring in the trace_source tests once we start emitting bytecode

    @requires_tlparse
    def test_graph_sizes_dynamic(self):
        def fn(a, b):
            return a @ b

        fn_opt = torch.compile(fn, backend="eager", dynamic=False)
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        fn_opt2 = torch.compile(fn, backend="eager", dynamic=True)
        fn_opt2(torch.randn(5, 10), torch.randn(10, 15))

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 800}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [10, 20], "is_leaf": true, "stride": [20, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 2400}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [20, 30], "is_leaf": true, "stride": [30, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['b']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [10, 20], "l_b_": [20, 30], "matmul": [10, 30]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 200}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [5, 10], "is_leaf": true, "stride": [10, 1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s0", "val": "5", "vr": "[2, int_oo]", "source": "L['a'].size()[0]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s1", "val": "10", "vr": "[2, int_oo]", "source": "L['a'].size()[1]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 600}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 2, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [10, 15], "is_leaf": true, "stride": [15, 1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['b']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s2", "val": "10", "vr": "[2, int_oo]", "source": "L['b'].size()[0]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"create_symbol": {"symbol": "s3", "val": "15", "vr": "[2, int_oo]", "source": "L['b'].size()[1]", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"guard_added_fast": {"expr": "Eq(s1, s2)", "user_stack": "STACK", "stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": ["s0", "s1"], "l_b_": ["s1", "s3"], "matmul": ["s0", "s3"]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    @requires_tlparse
    def test_guards_recompiles(self):
        def fn(x, ys, zs):
            return inner(x, ys, zs)

        def inner(x, ys, zs):
            for y, z in zip(ys, zs):
                x += y * z
            return x

        ys = [1.0, 2.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "x": [1]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"artifact": {"name": "recompile_reasons", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [1], "x": [1]}}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 1, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 1, "attempt": 0}
""",  # noqa: B950
        )

        self.assertParses()

    def test_dump_file(self):
        def f(x, y):
            return x.add(y)

        gm = fx.symbolic_trace(f)
        torch.compile(gm, backend="eager")(torch.randn(3), torch.randn(3))

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dump_file": {"name": "<eval_with_key>"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}



def forward(self, x, y):
    add = x.add(y);  x = y = None
    return add

{"describe_storage": {"id": 0, "describer_id": "ID", "size": 12}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [3], "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['x']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 1, "describer_id": "ID", "size": 12}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 1, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [3], "is_leaf": true, "stride": [1], "storage": 1, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 1, "source": "L['y']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_x_": [3], "l_y_": [3], "add": [3]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )

    @requires_tlparse
    @torch._inductor.config.patch("fx_graph_cache", True)
    def test_codecache(self):
        def fn(a):
            return a.sin()

        x = torch.tensor([1.0])
        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(x)
        torch._dynamo.reset()
        # Trigger a cache hit
        fn_opt(x)

        # Should print twice, including inductor_output_code
        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1], "sin": [1]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_runnable", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_post_grad_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_miss", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_start": {"stack": "STACK"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_storage": {"id": 0, "describer_id": "ID", "size": 4}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_tensor": {"id": 0, "ndim": 1, "dtype": "torch.float32", "device": "device(type='cpu')", "size": [1], "is_leaf": true, "stride": [1], "storage": 0, "view_func": "VIEW_FUNC", "describer_id": "ID"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"describe_source": {"describer_id": "ID", "id": 0, "source": "L['a']"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
{"dynamo_output_graph": {"sizes": {"l_a_": [1], "sin": [1]}}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aot_forward_graph_fw_metadata", "encoding": "string"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"aot_inference_graph": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"inductor_output_code": {"filename": "FILENAME"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "fx_graph_cache_hit", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"artifact": {"name": "aotautograd_cache_hash", "encoding": "json"}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"dynamo_cpp_guards_str": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}
{"compilation_metrics": "METRICS", "frame_id": 0, "frame_compile_id": 0, "attempt": 0}
""",  # noqa: B950
        )
        self.assertParses()

    @requires_tlparse
    def test_make_fx_fail_partial(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(StructuredTraceTestingFilter("make_fx_fail_partial"))
        trace_log.addHandler(payload_handler)

        def f(x):
            y = x + 1
            raise RuntimeError("boo")

        try:
            make_fx(f)(torch.randn(2))
        except RuntimeError:
            pass

        self.assertExpectedInline(
            self.buffer.getvalue(),
            """\
{"artifact": {"name": "make_fx_fail_partial", "encoding": "string"}, "stack": "STACK", "has_payload": "HASH"}
""",
        )

        self.assertExpectedInline(
            payload_buffer.getvalue(),
            """\
def forward(self, x_1: "f32[2][1]cpu"):
    # No stacktrace found for following nodes
    add: "f32[2][1]cpu" = torch.ops.aten.add.Tensor(x_1, 1);  x_1 = add = None
""",
        )

    @requires_tlparse
    @torch._inductor.config.patch("fx_graph_cache", True)
    @show_chrome_events
    def test_chromium_event(self):
        def fn(a):
            return a.sin()

        x = torch.tensor([1.0])
        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(x)
        torch._dynamo.reset()
        # Trigger a cache hit
        fn_opt(x)
        # Should print twice, including inductor_output_code
        self.assertParses()
        chromium_event = '{"chromium_event": {}, "frame_id": 0, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"}'
        self.assertTrue(chromium_event in self.buffer.getvalue())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
