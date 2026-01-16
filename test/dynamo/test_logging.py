# Owner(s): ["module: dynamo"]
import contextlib
import functools
import logging
import os
import re
import unittest
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.distributed as dist
from torch._dynamo.testing import (
    empty_line_normalizer,
    extract_graph_and_tracker,
    skipIfNotPy311,
)
from torch._dynamo.trace_rules import _as_posix_path
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    find_free_port,
    IS_WINDOWS,
    munge_exc,
    skipIfTorchDynamo,
    skipIfWindows,
    TEST_XPU,
    xfailIf,
)
from torch.testing._internal.inductor_utils import (
    HAS_CUDA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)
from torch.testing._internal.logging_utils import (
    LoggingTestCase,
    make_logging_test,
    make_settings_test,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


requires_gpu = unittest.skipUnless(
    HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON, "requires cuda or xpu with triton"
)

requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)

device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def munge_shape_guards(s: str) -> str:
    SHAPE_GUARD_REGEX = (
        r"[| ]* \+- SYMBOLIC_SHAPE_GUARD:"
        if torch._dynamo.config.enable_cpp_symbolic_shape_guards
        else r"^\+- LAMBDA_GUARD:"
    )

    def munge(s):
        s = re.sub(r"[^ ]+:\d+ in [^ ]+", "#:# in #", s)
        return re.subn(SHAPE_GUARD_REGEX, "+- __SHAPE_GUARD__:", s)

    lines = [munge(l) for l in s.splitlines()]
    return "\n".join([line for line, nsubs in lines if nsubs > 0])


def munge_global_state_json(text):
    import re

    match = re.search(r"\+- GLOBAL_STATE:.*", text)
    if not match:
        return ""

    line = match.group(0)
    while "[" in line:
        line = re.sub(r"\[[^\[\]]*\]", '"#"', line)

    line = re.sub(r':\s*(\d+|true|false|"[^"]*")', r': "#"', line)
    return line


LOG_PREFIX_PATTERNS = [
    re.compile(r"^\[rank\d+\]:\s*"),
    re.compile(r"^[A-Z]+:[^:]+:\s*"),
    re.compile(r"^[A-Z]\d{2,4}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?\s+\d+\s+[^\]]+\]\s*"),
    re.compile(r"^[A-Z](?:\d{4})?\s+[^:]+:\s*"),
]


def normalize_log_line(line: str) -> str:
    line = line.rstrip()
    for pattern in LOG_PREFIX_PATTERNS:
        stripped, count = pattern.subn("", line, count=1)
        if count:
            line = stripped.lstrip()
            break
    return line


def normalize_rank_prefix(output: str) -> str:
    if "[rank" in output:
        return output

    def repl(match):
        prefix = match.group(1)
        return f"{prefix}[rank0]: "

    return re.sub(r"(^|\n)(?:[A-Z]+:[^:]+:)", repl, output)


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    output = a.add(torch.ones(1000, 1000, device=device_type))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def multi_record_test(num_records, **kwargs):
    @make_logging_test(**kwargs)
    def fn(self, records):
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(*ARGS)
        self.assertEqual(len(records), num_records)

    return fn


def within_range_record_test(num_records_lower, num_records_higher, **kwargs):
    @make_logging_test(**kwargs)
    def fn(self, records):
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(*ARGS)
        self.assertGreaterEqual(len(records), num_records_lower)
        self.assertLessEqual(len(records), num_records_higher)

    return fn


def single_record_test(**kwargs):
    return multi_record_test(1, **kwargs)


class LoggingTests(LoggingTestCase):
    test_bytecode = multi_record_test(2, bytecode=True)
    test_output_code = multi_record_test(3, output_code=True)
    test_aot_graphs = multi_record_test(3, aot_graphs=True)

    @requires_gpu
    @make_logging_test(schedule=True)
    def test_schedule(self, records):
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 5)

    @requires_gpu
    @make_logging_test(fusion=True)
    def test_fusion(self, records):
        fn_opt = torch.compile(inductor_schedule_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertGreater(len(records), 0)

        # LOAF will add an extra round of fusion and result in more logs
        self.assertLess(
            len(records), 8 * (1 + torch._inductor.config.loop_ordering_after_fusion)
        )

    @requires_cuda_and_triton
    @make_logging_test(cudagraphs=True)
    def test_cudagraphs(self, records):
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device=device_type))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 8)

    @make_logging_test(recompiles=True)
    def test_recompiles(self, records):
        def outmost_fn(x, ys, zs):
            return outer_fn(x, ys, zs)

        def outer_fn(x, ys, zs):
            return fn(x, ys, zs)

        def fn(x, ys, zs):
            return inner(x, ys, zs)

        def inner(x, ys, zs):
            for y, z in zip(ys, zs):
                x += y * z
            return x

        ys = [1.0, 2.0, 3.0]
        zs = [3.0]
        x = torch.tensor([1.0])

        fn_opt = torch.compile(outmost_fn, backend="eager")
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        record_str = re.sub(
            r'"[^"]*"',
            "[file_path]",
            "\n".join(r.getMessage() for r in records),
        )
        self.assertIn(
            """\
    - User stack trace:
    -   File [file_path], line 201, in outmost_fn
    -     return outer_fn(x, ys, zs)
    -   File [file_path], line 204, in outer_fn
    -     return fn(x, ys, zs)
    -   File [file_path], line 207, in fn
    -     return inner(x, ys, zs)
    -   File [file_path], line 210, in inner
    -     for y, z in zip(ys, zs):""",
            record_str,
        )

    @make_logging_test(recompiles=True)
    def test_recompiles_closure_variable_hint(self, records):
        def make_cl(n1, n2):
            def inner(x):
                return x + n1 + n2

            return inner

        @torch.compile(backend="eager")
        def fn(cl, x):
            return cl(x)

        fn(make_cl(0, 1), torch.ones(3))
        fn(make_cl(0, 2), torch.ones(3))

        record_str = "\n".join(r.getMessage() for r in records)
        # The recompilation log should include a hint explaining which closure variable
        # the cell_contents refers to
        self.assertIn('(HINT: guard on "n2")', record_str)

    @make_logging_test(recompiles=True)
    def test_recompiles_nested_closure_variable_hint(self, records):
        # block_mask.mask_mod.__closure__[0].cell_contents.__closure__[0].cell_contents
        def make_inner_fn(inner_val):
            def inner(x):
                return x + inner_val

            return inner

        def make_outer_fn(outer_val, inner_fn):
            def outer(x):
                return inner_fn(x) * outer_val

            return outer

        class BlockMask:
            def __init__(self, mask_mod):
                self.mask_mod = mask_mod

        @torch.compile(backend="eager")
        def fn(block_mask, x):
            return block_mask.mask_mod(x)

        # inner_fn captures inner_val, outer captures (outer_val, inner_fn)
        # This creates: block_mask.mask_mod.__closure__[0].cell_contents.__closure__[0].cell_contents
        bm1 = BlockMask(make_outer_fn(2, make_inner_fn(10)))
        bm2 = BlockMask(make_outer_fn(2, make_inner_fn(20)))

        fn(bm1, torch.ones(3))
        fn(bm2, torch.ones(3))

        record_str = "\n".join(r.getMessage() for r in records)
        # The recompilation log should show the hint for the first closure variable
        self.assertIn('(HINT: guard on "inner_val")', record_str)
        # Verify it shows the full nested path
        self.assertIn("cell_contents.__closure__", record_str)

    @make_logging_test(recompiles=True)
    def test_recompiles_closure_variable_attribute_hint(self, records):
        # Test when guard is on an attribute of the closure cell contents
        # e.g., block_mask.mask_mod.__closure__[0].cell_contents.__code__
        # The hint should still show which closure variable is involved
        class Transform:
            def __init__(self, scale):
                self.scale = scale

            def __call__(self, x):
                return x * self.scale

        def make_fn(transform):
            def fn(x):
                return transform(x)

            return fn

        @torch.compile(backend="eager")
        def outer(inner_fn, x):
            return inner_fn(x)

        outer(make_fn(Transform(2)), torch.ones(3))
        outer(make_fn(Transform(3)), torch.ones(3))  # transform.scale changes

        record_str = "\n".join(r.getMessage() for r in records)
        # The hint should show the full path from closure var: "transform".scale
        self.assertIn('(HINT: guard on "transform".scale)', record_str)

    test_dynamo_debug = within_range_record_test(30, 90, dynamo=logging.DEBUG)
    test_dynamo_info = within_range_record_test(2, 10, dynamo=logging.INFO)

    @skipIfTorchDynamo("too slow")
    @make_logging_test(dynamo=logging.DEBUG)
    def test_dynamo_debug_default_off_artifacts(self, records):
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        self.assertEqual(len([r for r in records if ".__bytecode" in r.name]), 0)
        self.assertEqual(len([r for r in records if ".__output_code" in r.name]), 0)

    @make_logging_test(hierarchical_compile=True)
    def test_hierarchical_compile(self, records):
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def gn(x):
            return x * 2

        def fn(x):
            return gn(x)

        fn_opt = torch.compile(fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000))
        self.assertGreater(len(records), 0)

    @make_logging_test()
    def test_dynamo_error(self, records):
        try:
            fn_opt = torch.compile(dynamo_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        record = self.getRecord(records, "WON'T CONVERT")
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT dynamo_error_fn test_logging.py line N
due to:
Traceback (most recent call last):
torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_method add(*(FakeTensor(..., size=(1000, 1000), grad_fn=<MulBackward0>), FakeTensor(..., size=(10, 10))), **{}): got RuntimeError('Attempting to broadcast a dimension of length 10 at -1! Mismatching argument at index 1 had torch.Size([10, 10]); but expected shape should be broadcastable to [1000, 1000]')

from user code:
   File "test_logging.py", line N, in dynamo_error_fn
    output = output.add(torch.ones(10, 10))""",  # noqa: B950
        )

    test_aot = within_range_record_test(2, 6, aot=logging.INFO)
    test_inductor_debug = within_range_record_test(3, 33, inductor=logging.DEBUG)
    test_inductor_info = within_range_record_test(2, 10, inductor=logging.INFO)

    @make_logging_test()
    def test_inductor_error(self, records):
        exitstack = contextlib.ExitStack()
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError

        # inject an error in the lowerings
        dict_entries = {}
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                dict_entries[x] = throw

        exitstack.enter_context(
            unittest.mock.patch.dict(torch._inductor.lowering.lowerings, dict_entries)
        )

        try:
            fn_opt = torch.compile(inductor_error_fn, backend="inductor")
            fn_opt(*ARGS)
        except Exception:
            pass
        record = self.getRecord(records, "WON'T CONVERT")
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT inductor_error_fn test_logging.py line N
due to:
Traceback (most recent call last):
  File "test_logging.py", line N, in throw
    raise AssertionError
torch._inductor.exc.InductorError: LoweringException: AssertionError:
  target: aten.round.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='primals_1', layout=FixedLayout('cpu', torch.float32, size=[1000, 1000], stride=[1000, 1]))
  ))""",
        )

        exitstack.close()

    @requires_distributed()
    @requires_cuda_and_triton
    @make_logging_test(ddp_graphs=True)
    def test_ddp_graphs(self, records):
        class ToyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(1024, 1024),
                    torch.nn.Linear(1024, 1024),
                )

            def forward(self, x):
                return self.layers(x)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)

        model = DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        ddp_model = torch.compile(model, backend="inductor")

        ddp_model(torch.randn(1024, 1024, device="cuda:0"))

        dist.destroy_process_group()
        self.assertEqual(len([r for r in records if "__ddp_graphs" in r.name]), 4)

    # check that logging to a child log of a registered logger
    # does not register it and result in duplicated records
    @make_settings_test("torch._dynamo.output_graph")
    def test_open_registration_with_registered_parent(self, records):
        logger = logging.getLogger("torch._dynamo.output_graph")
        logger.info("hi")
        self.assertEqual(len(records), 1)

    # check logging to a random log that is not a child log of a registered
    # logger registers it and sets handlers properly
    @make_settings_test("torch.utils")
    def test_open_registration(self, records):
        logger = logging.getLogger("torch.utils")
        logger.info("hi")
        self.assertEqual(len(records), 1)

    # check logging to a random log that is not a child log of a registered
    # logger registers it and sets handlers properly
    @make_logging_test(modules={"torch.utils": logging.INFO})
    def test_open_registration_python_api(self, records):
        logger = logging.getLogger("torch.utils")
        logger.info("hi")
        self.assertEqual(len(records), 1)

    @make_settings_test("torch._logging")
    def test_directory_based_logging(self, records):
        # Test that the package itself can log
        logger = logging.getLogger("torch._logging")
        logger.info("package log")

        # Test that submodules can also log
        sublogger = logging.getLogger("torch._logging._internal")
        sublogger.info("submodule log")

        # We should have at least 2 records (one from package, one from submodule)
        self.assertGreaterEqual(len(records), 2)

        # Verify both loggers are registered and have handlers
        self.assertTrue(len(logger.handlers) > 0 or logger.propagate)
        self.assertTrue(len(sublogger.handlers) > 0 or sublogger.propagate)

    @make_logging_test(all=logging.DEBUG, dynamo=logging.INFO)
    def test_all(self, _):
        registry = torch._logging._internal.log_registry

        dynamo_qnames = registry.log_alias_to_log_qnames["dynamo"]
        for logger_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(logger_qname)

            # if logger_qname is a.b.c and dynamo_qnames contains a.b, it still matches dynamo's INFO setting
            if any(logger_qname.find(d) == 0 for d in dynamo_qnames):
                self.assertEqual(
                    logger.getEffectiveLevel(),
                    logging.INFO,
                    msg=f"expected {logger_qname} is INFO, got {logging.getLevelName(logger.getEffectiveLevel())}",
                )
            else:
                self.assertEqual(
                    logger.getEffectiveLevel(),
                    logging.DEBUG,
                    msg=f"expected {logger_qname} is DEBUG, got {logging.getLevelName(logger.getEffectiveLevel())}",
                )

    @make_logging_test(graph_breaks=True)
    def test_graph_breaks(self, records):
        @torch.compile(backend="inductor")
        def fn(x):
            torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(1))

        self.assertEqual(len(records), 1)

    @make_logging_test(side_effects=True)
    def test_side_effects(self, records):
        my_list = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x, lst):
            lst.append(4)
            return x + len(lst)

        fn(torch.ones(1), my_list)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type list (source name: L['lst'])

      File "test_logging.py", line N, in test_side_effects
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.append(4)
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_nested_calls(self, records):
        outer_list = [1]

        def inner(lst):
            lst.append(2)
            return len(lst)

        @torch.compile(backend="eager")
        def outer(x, my_list):
            result = inner(my_list)
            my_list.append(3)
            return x + result + len(my_list)

        outer(torch.ones(1), outer_list)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type list (source name: L['my_list'])

      File "test_logging.py", line N, in test_side_effects_nested_calls
        outer(torch.ones(1), outer_list)
      File "test_logging.py", line N, in outer
        result = inner(my_list)
      File "test_logging.py", line N, in inner
        lst.append(2)

    ********

      File "test_logging.py", line N, in test_side_effects_nested_calls
        outer(torch.ones(1), outer_list)
      File "test_logging.py", line N, in outer
        my_list.append(3)
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_multiple_mutations_same_object(self, records):
        my_list = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x, lst):
            lst.append(4)
            lst.append(5)
            lst.extend([6, 7])
            lst.pop()
            return x + len(lst)

        fn(torch.ones(1), my_list)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type list (source name: L['lst'])

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.append(4)

    ********

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.append(5)

    ********

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.extend([6, 7])

    ********

      File "test_logging.py", line N, in test_side_effects_multiple_mutations_same_object
        fn(torch.ones(1), my_list)
      File "test_logging.py", line N, in fn
        lst.pop()
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_dict_mutations(self, records):
        my_dict = {"a": 1}

        @torch.compile(backend="eager")
        def fn(x, d):
            d["b"] = 2
            d["c"] = 3
            return x + len(d)

        fn(torch.ones(1), my_dict)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type dict (source name: L['d'])

      File "test_logging.py", line N, in test_side_effects_dict_mutations
        fn(torch.ones(1), my_dict)
      File "test_logging.py", line N, in fn
        d["b"] = 2

    ********

      File "test_logging.py", line N, in test_side_effects_dict_mutations
        fn(torch.ones(1), my_dict)
      File "test_logging.py", line N, in fn
        d["c"] = 3
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_attribute_mutations(self, records):
        class MyClass:
            def __init__(self):
                self.value = 10
                self.count = 0

        obj = MyClass()

        @torch.compile(backend="eager")
        def fn(x, o):
            o.value = 20
            o.count = 1
            o.count = 2
            return x + o.value + o.count

        fn(torch.ones(1), obj)

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type MyClass (source name: L['o'])

      File "test_logging.py", line N, in test_side_effects_attribute_mutations
        fn(torch.ones(1), obj)
      File "test_logging.py", line N, in fn
        o.value = 20

    ********

      File "test_logging.py", line N, in test_side_effects_attribute_mutations
        fn(torch.ones(1), obj)
      File "test_logging.py", line N, in fn
        o.count = 1

    ********

      File "test_logging.py", line N, in test_side_effects_attribute_mutations
        fn(torch.ones(1), obj)
      File "test_logging.py", line N, in fn
        o.count = 2
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_local_list_no_log(self, records):
        """Test that lists created inside compiled region don't log side effects."""

        @torch.compile(backend="eager")
        def fn(x):
            my_list = [1, 2, 3]  # Created inside compiled region
            my_list.append(4)
            return x + len(my_list)

        fn(torch.ones(1))

        # Should NOT have logged the list mutation since it's a local variable
        self.assertEqual(len(records), 0)

    @make_logging_test(side_effects=True)
    def test_side_effects_local_object_with_log(self, records):
        """Test that returned objects created inside compiled region still log attribute mutations."""

        class MyClass:
            def __init__(self):
                self.value = 10

        @torch.compile(backend="eager")
        def fn(x):
            obj = MyClass()  # Created inside compiled region
            obj.value = 20
            return x + obj.value, obj

        fn(torch.ones(1))

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type MyClass (source: created in torch.compile region)

      File "test_logging.py", line N, in test_side_effects_local_object_with_log
        fn(torch.ones(1))
      File "test_logging.py", line N, in fn
        obj = MyClass()  # Created inside compiled region
      File "test_logging.py", line N, in __init__
        self.value = 10

    ********

      File "test_logging.py", line N, in test_side_effects_local_object_with_log
        fn(torch.ones(1))
      File "test_logging.py", line N, in fn
        obj.value = 20
""",
        )

    @make_logging_test(side_effects=True)
    def test_side_effects_nn_module_buffer(self, records):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.rand(2, 2))

            def forward(self, x):
                self.buf += 1
                return x + self.buf

        @torch.compile(backend="eager")
        def fn(mod, x):
            return mod(x)

        fn(Mod(), torch.ones(1))

        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage()),
            """\
Mutating object of type dict (source name: L['mod']._buffers)

      File "test_logging.py", line N, in test_side_effects_nn_module_buffer
        fn(Mod(), torch.ones(1))
      File "test_logging.py", line N, in fn
        return mod(x)
      File "test_logging.py", line N, in forward
        self.buf += 1
""",
        )

    @make_logging_test(side_effects=True)
    @torch._dynamo.config.patch(side_effect_replay_policy="silent")
    def test_side_effects_silent_config(self, records):
        my_list = [1, 2, 3]

        @torch.compile(backend="eager")
        def fn(x, lst):
            lst.append(4)
            return x + len(lst)

        fn(torch.ones(1), my_list)

        self.assertEqual(len(records), 0)

    @make_settings_test("torch._dynamo.utils")
    def test_dump_compile_times(self, records):
        fn_opt = torch.compile(example_fn, backend="inductor")
        fn_opt(torch.ones(1000, 1000))
        # This function runs during exit via atexit.register.
        # We're not actually going to run atexit._run_exit_funcs() here,
        # because it'll destroy state necessary for other tests.
        torch._dynamo.utils.dump_compile_times()
        self.assertEqual(
            len(
                [r for r in records if "TorchDynamo compilation metrics" in str(r.msg)]
            ),
            1,
        )

    @make_logging_test(dynamo=logging.INFO)
    def test_custom_format_exc(self, records):
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        try:
            raise RuntimeError("foo")
        except RuntimeError:
            dynamo_log.exception("test dynamo")
            dynamo_log.info("with exc", exc_info=True)
        dynamo_log.info("with stack", stack_info=True)
        self.assertEqual(len(records), 3)
        # unfortunately there's no easy way to test the final formatted log other than
        # to ask the dynamo logger's handler to format it.
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break
        self.assertIsNotNone(handler)
        self.assertIn("Traceback", handler.format(records[0]))
        self.assertIn("Traceback", handler.format(records[1]))
        self.assertIn("Stack", handler.format(records[2]))

    @make_logging_test(dynamo=logging.INFO)
    def test_custom_format(self, records):
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        test_log = torch._logging.getArtifactLogger(
            torch._dynamo.__name__, "custom_format_test_artifact"
        )
        dynamo_log.info("test dynamo")
        test_log.info("custom format")
        self.assertEqual(len(records), 2)
        # unfortunately there's no easy way to test the final formatted log other than
        # to ask the dynamo logger's handler to format it.
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break
        self.assertIsNotNone(handler)
        formatted_dynamo = handler.format(records[0])
        self.assertIn("test dynamo", formatted_dynamo)
        self.assertEqual(normalize_log_line(formatted_dynamo), "test dynamo")
        ci_style_line = (
            "I1124 19:43:23.879000 4928 dynamo/test_logging.py:410] test dynamo"
        )
        self.assertEqual(normalize_log_line(ci_style_line), "test dynamo")

        formatted_artifact = handler.format(records[1])
        self.assertIn("custom format", formatted_artifact)
        self.assertEqual(normalize_log_line(formatted_artifact), "custom format")

    @make_logging_test(dynamo=logging.INFO)
    def test_multiline_format(self, records):
        dynamo_log = logging.getLogger(torch._dynamo.__name__)
        dynamo_log.info("test\ndynamo")
        dynamo_log.info("%s", "test\ndynamo")
        dynamo_log.info("test\n%s", "test\ndynamo")
        self.assertEqual(len(records), 3)
        # unfortunately there's no easy way to test the final formatted log other than
        # to ask the dynamo logger's handler to format it.
        for handler in dynamo_log.handlers:
            if torch._logging._internal._is_torch_handler(handler):
                break
        self.assertIsNotNone(handler)
        expected_lines = [
            ["test", "dynamo"],
            ["test", "dynamo"],
            ["test", "test", "dynamo"],
        ]

        for record, expected in zip(records, expected_lines):
            formatted = handler.format(record)
            normalized_lines = [
                line
                for line in (normalize_log_line(l) for l in formatted.splitlines())
                if line
            ]
            self.assertEqual(normalized_lines, expected)

    test_trace_source_simple = within_range_record_test(1, 100, trace_source=True)

    @make_logging_test(trace_source=True)
    def test_trace_source_if_stmt(self, records):
        def fn(x):
            if x.sum() > 0:
                return x * 2
            return x * 3

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.ones(3, 3))

        found_x2 = False
        found_x3 = False
        for record in records:
            msg = record.getMessage()
            if "return x * 2" in msg:
                found_x2 = True
            if "return x * 3" in msg:
                found_x3 = True

        self.assertTrue(found_x2)
        self.assertFalse(found_x3)

    @make_logging_test(trace_source=True)
    def test_trace_source_nested(self, records):
        def fn1(x):
            x = fn2(x)
            return x * 2

        def fn2(x):
            x = fn3(x)
            return x * 3

        def fn3(x):
            return x * 4

        fn_opt = torch.compile(fn1, backend="eager")
        fn_opt(torch.ones(3, 3))

        found_x2 = False
        found_x3 = False
        found_x4 = False
        for record in records:
            msg = record.getMessage()
            if "return x * 2" in msg:
                found_x2 = True
                self.assertNotIn("inline depth", msg)
            elif "return x * 3" in msg:
                found_x3 = True
                self.assertIn("inline depth: 1", msg)
            elif "return x * 4" in msg:
                found_x4 = True
                self.assertIn("inline depth: 2", msg)
        self.assertTrue(found_x2)
        self.assertTrue(found_x3)
        self.assertTrue(found_x4)

    @make_logging_test(trace_source=True)
    def test_trace_source_cond(self, records):
        from functorch.experimental.control_flow import cond

        def true_fn(x):
            return x * 2

        def false_fn(x):
            return x * 3

        def inner(pred, x):
            return cond(pred, true_fn, false_fn, [x])

        def outer(pred, x):
            return inner(pred, x)

        fn_opt = torch.compile(outer, backend="eager")
        fn_opt(torch.tensor(True), torch.ones(3, 3))

        found_x2 = False
        found_x3 = False
        for record in records:
            msg = record.getMessage()
            if "return x * 2" in msg:
                found_x2 = True
                self.assertIn("inline depth: 3", msg)
            if "return x * 3" in msg:
                found_x3 = True
                self.assertIn("inline depth: 3", msg)

        self.assertTrue(found_x2)
        self.assertTrue(found_x3)

    @make_logging_test(trace_source=True)
    def test_trace_source_funcname(self, records):
        # NOTE: list comprehensions are inlined in 3.12, so test with tuples
        def fn1():
            def fn2():
                if True:
                    return tuple(torch.ones(3, 3) for _ in range(5))
                return None

            return fn2()

        fn_opt = torch.compile(fn1, backend="eager")
        fn_opt()

        found_funcname = False
        for record in records:
            msg = record.getMessage()
            if "<genexpr>" in msg and "fn1.fn2" in msg:
                found_funcname = True

        self.assertTrue(found_funcname)

    def test_invalid_artifact_flag(self):
        with self.assertRaises(ValueError):
            torch._logging.set_logs(aot_graphs=5)

    def test_invalid_artifact_flag_error_msg(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "not_an_existing_log_artifact_should_error"
        _, stderr = self.run_process_no_exception(
            "import torch",
            env=env,
        )
        lines = stderr.decode().split("\r\n" if IS_WINDOWS else "\n")
        # This is a sanity assert that our error is not spammy.
        # As of this test creation this was 18.
        # See this issue for the purpose o this test:
        # https://github.com/pytorch/pytorch/issues/151055
        self.assertTrue(len(lines) < 50)
        # The other sanity assert - check that the last few lines
        # map to the actual error message we want to raise
        # (I could use an expecttest here, although it would break
        #  whenever someone adds a new logging artifact)
        self.assertEqual(
            lines[-5], 'For more info on various settings, try TORCH_LOGS="help"'
        )
        self.assertEqual(lines[-4], "Valid settings:")

    @requires_distributed()
    @skipIfWindows(msg="TODO: (xuhancn), Can't reproduce locally")
    def test_distributed_rank_logging(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "dynamo"
        _, stderr = self.run_process_no_exception(
            """\
import torch.distributed as dist
import logging
from torch.testing._internal.distributed.fake_pg import FakeStore
store = FakeStore()
dist.init_process_group("fake", rank=0, world_size=2, store=store)
dynamo_log = logging.getLogger("torch._dynamo")
dynamo_log.info("woof")
print("arf")
""",
            env=env,
        )
        stderr_text = stderr.decode("utf-8")
        normalized = normalize_rank_prefix(stderr_text)
        self.assertIn("[rank0]:", normalized)
        self.assertIn("woof", normalized)

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call(self, records):
        def fn(x, y):
            return (x * 2) @ (y * 3)

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        self.assertEqual(len(records), 3)
        # only get last 2 lines
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        self.assertExpectedInline(
            messages[0],
            """\
            return (x * 2) @ (y * 3)
                    ~~^~~""",
        )
        self.assertExpectedInline(
            messages[1],
            """\
            return (x * 2) @ (y * 3)
                              ~~^~~""",
        )
        self.assertExpectedInline(
            messages[2],
            """\
            return (x * 2) @ (y * 3)
                   ~~~~~~~~^~~~~~~~~""",
        )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_prefix(self, records):
        def fn(x, y):
            return (x * 2) @ (y * 3)

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.randn(10, 20), torch.randn(20, 30))

        msg0 = munge_exc(records[0].getMessage())
        self.assertExpectedInline(
            msg0,
            """\
TRACE FX call mul from test_logging.py:N in fn (LoggingTests.test_trace_call_prefix.fn)
            return (x * 2) @ (y * 3)
                    ~~^~~""",
        )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_inline_call(self, records):
        def g(x):
            return x * 2

        def f(x):
            return g(g(x))

        fn_opt = torch.compile(f, backend="eager")
        fn_opt(torch.randn(3, 3))

        self.assertEqual(len(records), 3)
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        self.assertExpectedInline(
            messages[0],
            """\
            return g(g(x))
                     ~^^^""",
        )
        self.assertExpectedInline(
            messages[1],
            """\
            return x * 2
                   ~~^~~""",
        )
        # skip this check since 3.13 removed carets for this case
        # see https://github.com/python/cpython/issues/99180
        # self.assertExpectedInline(
        #     messages[2],
        #     """\
        #     return g(g(x))
        #            ~^^^^^^""",
        # )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_graph_break(self, records):
        def fn(x):
            x = x * 2
            torch._dynamo.graph_break()
            return x * 3

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.randn(3, 3))

        self.assertEqual(len(records), 3)
        messages = [
            "\n".join(record.getMessage().split("\n")[-2:]) for record in records
        ]
        self.assertExpectedInline(
            messages[0],
            """\
            x = x * 2
                ~~^~~""",
        )
        self.assertExpectedInline(
            messages[-1],
            """\
            return x * 3
                   ~~^~~""",
        )

    @make_logging_test(guards=True, recompiles=True)
    def test_guards_recompiles(self, records):
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

        record_str = "\n".join(r.getMessage() for r in records)

        self.assertIn(
            """L['zs'][0] == 3.0""",
            record_str,
        )
        self.assertIn(
            "len(L['ys']) == 2",
            record_str,
        )

    @make_logging_test(guards=True)
    def test_guards_sloc(self, records):
        @torch.compile(dynamic=True, backend="eager")
        def f(x, y, z):
            x = x * 3
            if x.size(0) % 3 == 0:
                return x + torch.cat([y, z])
            else:
                return x * 2

        f(torch.randn(6), torch.randn(3), torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_shape_guards(record.getMessage()),
            """\
+- __SHAPE_GUARD__: L['x'].size()[0] == 2*L['y'].size()[0]  # return x + torch.cat([y, z])  # #:# in # #:# in #
+- __SHAPE_GUARD__: L['z'].size()[0] == L['y'].size()[0]  # duck sizing added this equality because these variables had the same size 3 (to avoid this specialization, set torch.fx.experimental._config.use_duck_shape = False)
+- __SHAPE_GUARD__: ((2*L['y'].size()[0]) % 3) == 0  # if x.size(0) % 3 == 0:  # #:# in # #:# in #
+- __SHAPE_GUARD__: 2 <= L['y'].size()[0]  # return x + torch.cat([y, z])  # #:# in # (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))""",  # noqa: B950
        )

    @make_logging_test(guards=True)
    def test_guards_polyfill_sloc(self, records):
        @torch.compile(dynamic=True, backend="eager")
        def f(x, y):
            return any([x.size(0) == y.size(0) * 2])

        f(torch.randn(6), torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_shape_guards(record.getMessage()),
            """\
+- __SHAPE_GUARD__: L['x'].size()[0] == 2*L['y'].size()[0]  # return any([x.size(0) == y.size(0) * 2])  # #:# in # #:# in #
+- __SHAPE_GUARD__: 2 <= L['y'].size()[0]  # return any([x.size(0) == y.size(0) * 2])  # #:# in # (user code shown is first use of this value--the guard itself is not due user code but due to 0/1 specialization in the framework; to avoid specialization try torch._dynamo.decorators.mark_unbacked(tensor, dim))""",  # noqa: B950
        )

    @make_logging_test(guards=True)
    def test_guards_sloc_vr(self, records):
        @torch.compile(dynamic=True, backend="eager")
        def f(x, y):
            torch._check(x.size(0) > 5)
            torch._check(x.size(0) < 30)
            torch._check(x.size(0) == y.size(0) * 2)
            return torch.tensor(True)

        f(torch.randn(6), torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_shape_guards(record.getMessage()),
            """\
+- __SHAPE_GUARD__: L['x'].size()[0] == 2*L['y'].size()[0]  # torch._check(x.size(0) == y.size(0) * 2)  # #:# in # #:# in #
+- __SHAPE_GUARD__: 3 <= L['y'].size()[0] <= 14  # torch._check(x.size(0) > 5)  # #:# in # #:# in # and torch._check(x.size(0) < 30)  # #:# in # #:# in #""",  # noqa: B950
        )

    @make_logging_test(guards=True)
    def test_global_state_guard_logging(self, records):
        @torch.compile(backend="eager")
        def f(x):
            return x + 1

        f(torch.randn(3))

        record = self.getRecord(records, "TREE_GUARD_MANAGER")
        self.assertExpectedInline(
            munge_global_state_json(record.getMessage()),
            """+- GLOBAL_STATE: ___check_global_state() against {"allow_bf16_reduce": "#","allow_fp16_reduce": "#","allow_tf32": "#","autocast_state":{"cached_enabled": "#","dtype": "#","enabled": "#"},"default_dtype": "#","deterministic_algorithms": "#","deterministic_algorithms_warn_only": "#","grad_mode": "#","num_threads": "#","torch_function": "#","torch_function_all_disabled": "#"}""",  # noqa: B950
        )

    @make_logging_test(cudagraph_static_inputs=True)
    def test_cudagraph_static_inputs(self, records):
        @torch.compile(mode="reduce-overhead")
        def fn(x):
            return x + 1

        x = torch.ones(2, 2)
        torch._dynamo.mark_static_address(x)
        fn(x)
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 4)

    @xfailIf(TEST_XPU)  # https://github.com/pytorch/pytorch/issues/157778
    @make_logging_test(perf_hints=True)
    @requires_gpu
    def test_optimizer_non_static_param(self, records):
        params = [torch.randn(10, 10, device=device_type) for _ in range(2)]
        for param in params:
            param.grad = torch.zeros_like(param)
        opt = torch.optim.Adam(params)
        compiled_opt_step = torch.compile(opt.step, mode="reduce-overhead")
        compiled_opt_step()
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 3)

    @make_logging_test(autotuning=True)
    @requires_gpu
    @unittest.skipIf(not SM90OrLater, "requires H100+ GPU")
    def test_autotuning(self, records):
        with torch._inductor.utils.fresh_cache():

            def f(a, b):
                return torch.mm(a, b)

            f = torch.compile(f, mode="max-autotune-no-cudagraphs")
            f(
                torch.randn(10, 10, device=device_type),
                torch.randn(10, 10, device=device_type),
            )
            self.assertGreater(len(records), 0)
            self.assertLess(len(records), 40)

    @make_logging_test(graph_region_expansion=True)
    def test_graph_region_expansion(self, records):
        with torch._dynamo.config.patch("track_nodes_for_deduplication", True):

            def inner_fn(x, y):
                x0 = x + 1
                y0 = y + 2
                z = x0.sum() + y0.sum()
                return z

            def fn(x, y):
                o0 = inner_fn(x, y)
                o1 = torch.sin(o0)
                o2 = inner_fn(x, o1)
                o3 = inner_fn(x, y)
                return o2 * o3 * o3

            graph, tracker = extract_graph_and_tracker(
                fn, torch.randn(10, 10), torch.randn(10, 10)
            )
            tracker.get_identical_regions(graph)
            self.assertGreater(len(records), 0)

    @skipIfTorchDynamo("too slow")
    @make_logging_test(**torch._logging.DEFAULT_LOGGING)
    def test_default_logging(self, records):
        def fn(a):
            if a.sum() < 0:
                a = torch.sin(a)
            else:
                a = torch.cos(a)
            print("hello")
            return a + 1

        fn_opt = torch.compile(fn, backend="eager")
        fn_opt(torch.ones(10, 10))
        fn_opt(-torch.ones(10, 5))

        self.assertGreater(len([r for r in records if ".__graph_breaks" in r.name]), 0)
        self.assertGreater(len([r for r in records if ".__recompiles" in r.name]), 0)
        self.assertGreater(len([r for r in records if ".symbolic_shapes" in r.name]), 0)
        self.assertGreater(len([r for r in records if ".__guards" in r.name]), 0)
        self.assertGreater(
            len([r for r in records if "return a + 1" in r.getMessage()]), 0
        )

    def test_logs_out(self):
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            file_path = _as_posix_path(tmp.name)
            """
            NamedTemporaryFile will include a file open operation.
            On Windowsm the file is opened by NamedTemporaryFile, the
            following run_process_no_exception can't access a opened file.
            And then, raise a PermissionError: [Errno 13] Permission denied: [file_path]
            """
            tmp.close()
            env = dict(os.environ)
            env["TORCH_LOGS"] = "dynamo"
            env["TORCH_LOGS_OUT"] = file_path
            _, stderr = self.run_process_no_exception(
                """\
import torch
@torch.compile(backend="eager")
def fn(a):
    return a.sum()

fn(torch.randn(5))
                """,
                env=env,
            )
            with open(
                file_path, encoding="utf-8"
            ) as fd:  # encoding file to UTF-8 for Windows.
                lines = fd.read()
                orig_maxDiff = unittest.TestCase.maxDiff
                unittest.TestCase.maxDiff = None
                try:
                    self.assertEqual(  # process wrap difference: /r/n on Windows, /n on posix.
                        empty_line_normalizer(lines),
                        empty_line_normalizer(stderr.decode("utf-8")),
                    )
                except Exception:
                    unittest.TestCase.maxDiff = orig_maxDiff
                    raise

    @make_settings_test("torch._dynamo.eval_frame")
    def test_log_traced_frames(self, records):
        torch._dynamo.eval_frame.clear_dynamo_tls()

        # Test program
        @torch.compile()
        def foo():
            x = torch.ones([10])

            def bar():
                y = x + x
                torch._dynamo.graph_break()
                z = y * x
                return z

            return bar(), bar

        foo()

        # `_log_traced_frames` is registered as an atexit callback, so we invoke
        # it explicitly for testing.
        torch._dynamo.eval_frame._log_traced_frames()

        # Get the relevant log.
        record = self.getRecord(records, "TorchDynamo attempted to trace")

        # Check
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
TorchDynamo attempted to trace the following frames: [
  * foo test_logging.py:N
  * bar test_logging.py:N
]""",
        )


# non single record tests
exclusions = {
    "bytecode",
    "cudagraphs",
    "output_code",
    "schedule",
    "fusion",
    "overlap",
    "aot_graphs",
    "aot_graphs_effects",
    "pre_grad_graphs",
    "joint_graph_passes",
    "post_grad_graphs",
    "inductor_metrics",
    "ir_pre_fusion",
    "ir_post_fusion",
    "compiled_autograd",
    "compiled_autograd_verbose",
    "recompiles",
    "recompiles_verbose",
    "graph_breaks",
    "side_effects",
    "graph",
    "graph_code",
    "graph_code_verbose",
    "graph_sizes",
    "ddp_graphs",
    "perf_hints",
    "not_implemented",
    "trace_source",
    "trace_call",
    "trace_bytecode",
    "custom_format_test_artifact",
    "onnx",
    "onnx_diagnostics",
    "guards",
    "verbose_guards",
    "sym_node",
    "export",
    "trace_shape_events",
    "cudagraph_static_inputs",
    "benchmarking",
    "loop_ordering",
    "loop_tiling",
    "auto_chunker",
    "autotuning",
    "graph_region_expansion",
    "hierarchical_compile",
    "compute_dependencies",
    "annotation",
    "node_runtime_estimation",
    "caching",
}
for name in torch._logging._internal.log_registry.artifact_names:
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(**{name: True}))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
