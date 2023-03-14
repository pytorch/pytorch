# Owner(s): ["module: dynamo"]
import functools
import unittest.mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging.loggable_types as rec_types
from torch.testing._internal.dynamo_logging_utils import make_logging_test

from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.testing._internal.logging_utils import LoggingTestCase

requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, "requires cuda")


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
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


def multi_record_test(name, ty, num_records):
    @make_logging_test(name)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        self.assertEqual(len(records), num_records)
        self.assertIsInstance(records[0].msg, ty)

    return fn


def within_range_record_test(name, ty, num_records_lower, num_records_higher):
    @make_logging_test(name)
    def fn(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(example_fn)
        fn_opt(*ARGS)
        self.assertGreaterEqual(len(records), num_records_lower)
        self.assertLessEqual(len(records), num_records_higher)
        self.assertIsInstance(records[0].msg, ty)

    return fn


def single_record_test(name, ty):
    return multi_record_test(name, ty, 1)


class LoggingTests(LoggingTestCase):

    test_bytecode = multi_record_test("bytecode", rec_types.ByteCodeLogRec, 2)
    test_output_code = multi_record_test("output_code", rec_types.OutputCodeLogRec, 2)

    @requires_cuda()
    @make_logging_test("schedule")
    def test_schedule(self, records):
        fn_opt = torch._dynamo.optimize("inductor")(inductor_schedule_fn)
        fn_opt(torch.ones(1000, 1000, device="cuda"))
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, rec_types.ScheduleLogRec)

    test_dynamo_debug = within_range_record_test("+dynamo", str, 30, 50)
    test_dynamo_info = within_range_record_test("dynamo", str, 2, 10)

    @make_logging_test("-dynamo")
    def test_dynamo_error(self, records):
        try:
            fn_opt = torch._dynamo.optimize("inductor")(dynamo_error_fn)
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, str)

    test_aot = multi_record_test("aot", rec_types.AOTJointGraphLogRec, 3)
    test_inductor_debug = within_range_record_test("+inductor", str, 5, 15)
    test_inductor_info = within_range_record_test("inductor", str, 2, 4)

    @make_logging_test("-inductor")
    def test_inductor_error(self, records):
        import torch._inductor.lowering

        def throw(x):
            raise AssertionError()

        # inject an error in the lowerings
        for x in list(torch._inductor.lowering.lowerings.keys()):
            if "round" in x.__name__:
                torch._inductor.lowering.lowerings[x] = throw

        try:
            fn_opt = torch._dynamo.optimize("inductor")(inductor_error_fn)
            fn_opt(*ARGS)
        except Exception:
            pass
        self.assertEqual(len(records), 1)
        self.assertIsInstance(records[0].msg, str)


# single record tests
exclusions = {"bytecode", "output_code", "schedule"}
for name, ty in torch._logging.internal.log_registry.name_to_rec_type.items():
    if name not in exclusions:
        setattr(LoggingTests, f"test_{name}", single_record_test(name, ty))

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
