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

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import (
    LoggingTestCase,
    make_logging_test,
    make_settings_test,
)

requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")
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
    test_aot_graphs = multi_record_test(2, aot_graphs=True)

    @requires_cuda()
    @make_logging_test(schedule=True)
    def test_schedule(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertGreater(len(records), 0)
        self.assertLess(len(records), 5)

    @make_logging_test(recompiles=True)
    def test_recompiles(self, records):
        def fn(x, y):
            return torch.add(x, y)

        fn_opt = torch._dynamo.optimize("inductor")(fn)
        fn_opt(torch.ones(1000, 1000), torch.ones(1000, 1000))
        fn_opt(torch.ones(1000, 1000), 1)
        self.assertGreater(len(records), 0)

    test_dynamo_debug = within_range_record_test(30, 50, dynamo=logging.DEBUG)
    test_dynamo_info = within_range_record_test(2, 10, dynamo=logging.INFO)

    @make_logging_test(dynamo=logging.DEBUG)
    def test_dynamo_debug_no_bytecode(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(torch.ones(1000, 1000))
        self.assertEqual(len([r for r in records if ".__bytecode" in r.name]), 0)

    @make_logging_test(dynamo=logging.ERROR)
    def test_dynamo_error(self, records):
        try:
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertEqual(len(records), 1)

    test_aot = within_range_record_test(2, 6, aot=logging.INFO)
    test_inductor_debug = within_range_record_test(3, 15, inductor=logging.DEBUG)
    test_inductor_info = within_range_record_test(2, 4, inductor=logging.INFO)

    @make_logging_test(dynamo=logging.ERROR)
    def test_inductor_error(self, records):
        exitstack = contextlib.ExitStack()
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError()

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
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, str)

        exitstack.close()

    @requires_distributed()
    @requires_cuda()
    @make_logging_test(ddp_graphs=True)
    def test_ddp_graphs(self, records):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super(ToyModel, self).__init__()
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
        state = torch._logging._internal.log_state

        dynamo_qname = registry.log_alias_to_log_qname["dynamo"]
        for logger_qname in torch._logging._internal.log_registry.get_log_qnames():
            logger = logging.getLogger(logger_qname)

            if logger_qname == dynamo_qname:
                self.assertEqual(logger.level, logging.INFO)
            else:
                self.assertEqual(logger.level, logging.DEBUG)


# single record tests
exclusions = {
    "bytecode",
    "output_code",
    "schedule",
    "aot_graphs",
    "recompiles",
    "ddp_graphs",
    "perf_hints",
    "not_implemented",
}
for name in torch._logging._internal.log_registry.artifact_names:
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(**{name: True}))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
