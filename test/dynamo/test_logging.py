# Owner(s): ["module: dynamo"]
import contextlib
import functools
import logging
import os
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.distributed as dist
from torch._dynamo.testing import skipIfNotPy311

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.testing._internal.common_utils import (
    find_free_port,
    munge_exc,
    skipIfTorchDynamo,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import (
    LoggingTestCase,
    make_logging_test,
    make_settings_test,
)

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


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
    output = a.add(torch.ones(1000, 1000, device="cuda"))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def multi_record_test(num_records, **kwargs):
    @make_logging_test(**kwargs)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        self.assertEqual(len(records), num_records)

    return fn


def within_range_record_test(num_records_lower, num_records_higher, **kwargs):
    @make_logging_test(**kwargs)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        self.assertGreaterEqual(len(records), num_records_lower)
        self.assertLessEqual(len(records), num_records_higher)

    return fn


def single_record_test(**kwargs):
    return multi_record_test(1, **kwargs)


class LoggingTests(LoggingTestCase):
    test_bytecode = multi_record_test(2, bytecode=True)
    test_output_code = multi_record_test(2, output_code=True)
    test_aot_graphs = multi_record_test(3, aot_graphs=True)

    @requires_cuda
    @make_logging_test(schedule=True)
    def test_schedule(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 5)

    @requires_cuda
    @make_logging_test(fusion=True)
    def test_fusion(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 8)

    @requires_cuda
    @make_logging_test(cudagraphs=True)
    def test_cudagraphs(self, records):
        fn_opt = torch.compile(mode="reduce-overhead")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 8)

    @make_logging_test(recompiles=True)
    def test_recompiles(self, records):
        def fn(x, y):
            return torch.add(x, y)

        fn_opt = torch._dynamo.optimize("inductor")(fn)
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)
        self.assertGreater(len(records), 0)

    test_dynamo_debug = within_range_record_test(30, 90, dynamo=logging.DEBUG)
    test_dynamo_info = within_range_record_test(2, 10, dynamo=logging.INFO)

    @skipIfTorchDynamo("too slow")
    @make_logging_test(dynamo=logging.DEBUG)
    def test_dynamo_debug_default_off_artifacts(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(torch.ones(1000, 1000))
        self.assertEqual(len([r for r in records if ".__bytecode" in r.name]), 0)
        self.assertEqual(len([r for r in records if ".__output_code" in r.name]), 0)

    @make_logging_test()
    def test_dynamo_error(self, records):
        try:
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
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
torch._dynamo.exc.TorchRuntimeError: Failed running call_method add(*(FakeTensor(..., size=(1000, 1000), grad_fn=<MulBackward0>), FakeTensor(..., size=(10, 10))), **{}):
Attempting to broadcast a dimension of length 10 at -1! Mismatching argument at index 1 had torch.Size([10, 10]); but expected shape should be broadcastable to [1000, 1000]

from user code:
   File "test_logging.py", line N, in dynamo_error_fn
    output = output.add(torch.ones(10, 10))""",  # noqa: B950
        )

    test_aot = within_range_record_test(2, 6, aot=logging.INFO)
    test_inductor_debug = within_range_record_test(3, 17, inductor=logging.DEBUG)
    test_inductor_info = within_range_record_test(2, 4, inductor=logging.INFO)

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
            fn_opt = torch._dynamo.optimize("inductor")(inductor_error_fn)
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
torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
LoweringException: AssertionError:
  target: aten.round.default
  args[0]: TensorBox(StorageBox(
    InputBuffer(name='primals_1', layout=FixedLayout('cpu', torch.float32, size=[1000, 1000], stride=[1000, 1]))
  ))""",
        )

        exitstack.close()

    @requires_distributed()
    @requires_cuda
    @make_logging_test(ddp_graphs=True)
    def test_ddp_graphs(self, records):
        class ToyModel(torch.nn.Module):
            def __init__(self):
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

        ddp_model = torch._dynamo.optimize("inductor")(
            DDP(ToyModel().to("cuda:0"), device_ids=[0], bucket_cap_mb=4)
        )

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
        @torch._dynamo.optimize("inductor")
        def fn(x):
            torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(1))

        self.assertEqual(len(records), 1)

    @make_settings_test("torch._dynamo.utils")
    def test_dump_compile_times(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
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
        self.assertIn("I", handler.format(records[0]))
        self.assertEqual("custom format", handler.format(records[1]))

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
        for record in records:
            r = handler.format(record)
            for l in r.splitlines():
                self.assertIn("I", l)

    test_trace_source_simple = within_range_record_test(1, 100, trace_source=True)

    @make_logging_test(trace_source=True)
    def test_trace_source_if_stmt(self, records):
        def fn(x):
            if x.sum() > 0:
                return x * 2
            return x * 3

        fn_opt = torch._dynamo.optimize("eager")(fn)
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

        fn_opt = torch._dynamo.optimize("eager")(fn1)
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

        fn_opt = torch._dynamo.optimize("eager")(outer)
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

        fn_opt = torch._dynamo.optimize("eager")(fn1)
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

    @requires_distributed()
    def test_distributed_rank_logging(self):
        env = dict(os.environ)
        env["TORCH_LOGS"] = "dynamo"
        stdout, stderr = self.run_process_no_exception(
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
        self.assertIn("[rank0]:", stderr.decode("utf-8"))

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call(self, records):
        def fn(x, y):
            return (x * 2) @ (y * 3)

        fn_opt = torch._dynamo.optimize("eager")(fn)
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
    def test_trace_call_inline_call(self, records):
        def g(x):
            return x * 2

        def f(x):
            return g(g(x))

        fn_opt = torch._dynamo.optimize("eager")(f)
        fn_opt(torch.randn(3, 3))

        self.assertEqual(len(records), 4)
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
        self.assertExpectedInline(
            messages[2],
            """\
            return g(g(x))
                   ~^^^^^^""",
        )
        self.assertExpectedInline(
            messages[3],
            """\
            return x * 2
                   ~~^~~""",
        )

    @skipIfNotPy311
    @make_logging_test(trace_call=True)
    def test_trace_call_graph_break(self, records):
        def fn(x):
            x = x * 2
            torch._dynamo.graph_break()
            return x * 3

        fn_opt = torch._dynamo.optimize("eager")(fn)
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

        fn_opt = torch._dynamo.optimize("eager")(fn)
        fn_opt(x, ys, zs)
        fn_opt(x, ys[:1], zs)

        record_str = "\n".join(r.getMessage() for r in records)

        self.assertIn(
            """\
L['zs'][0] == 3.0                                             # for y, z in zip(ys, zs):""",
            record_str,
        )
        self.assertIn(
            """\
    triggered by the following guard failure(s):\n\
    - len(L['ys']) == 2                                             # for y, z in zip(ys, zs):""",
            record_str,
        )

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

        fn_opt = torch._dynamo.optimize("eager")(fn)
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

        with tempfile.NamedTemporaryFile() as tmp:
            env = dict(os.environ)
            env["TORCH_LOGS"] = "dynamo"
            env["TORCH_LOGS_OUT"] = tmp.name
            stdout, stderr = self.run_process_no_exception(
                """\
import torch
@torch.compile(backend="eager")
def fn(a):
    return a.sum()

fn(torch.randn(5))
                """,
                env=env,
            )
            with open(tmp.name) as fd:
                lines = fd.read()
                self.assertEqual(lines, stderr.decode("utf-8"))


# single record tests
exclusions = {
    "bytecode",
    "cudagraphs",
    "output_code",
    "schedule",
    "fusion",
    "overlap",
    "aot_graphs",
    "post_grad_graphs",
    "compiled_autograd",
    "compiled_autograd_verbose",
    "recompiles",
    "recompiles_verbose",
    "graph_breaks",
    "graph",
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
}
for name in torch._logging._internal.log_registry.artifact_names:
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(**{name: True}))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
