# Owner(s): ["module: dynamo"]
import io
import logging
import warnings
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import same
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


logger = logging.getLogger(__name__)
logger_test = logging.getLogger("test")


def f_info(x):
    x = x + x
    logger.info("moo")
    x = x * x
    return x


def f_isEnabledFor(x):
    x = x + x
    if logger.isEnabledFor(logging.INFO):
        logger.info("moo")
    x = x * x
    return x


@instantiate_parametrized_tests
class IgnoreLogsTests(torch._dynamo.test_case.TestCase):
    @parametrize(
        "ignore_method, fn, should_ignore_logger",
        [
            (None, f_info, False),
            (logger_test.info, f_info, False),
            (None, f_isEnabledFor, False),
            (logger_test.isEnabledFor, f_isEnabledFor, False),
            (logger.info, f_info, True),
            (logging.Logger.info, f_info, True),
            (logger.isEnabledFor, f_isEnabledFor, True),
            (logging.Logger.isEnabledFor, f_isEnabledFor, True),
        ],
    )
    def test_ignore_logger(self, ignore_method, fn, should_ignore_logger):
        counters.clear()
        x = torch.randn(3, 3)
        orig_out = fn(x)
        with torch._dynamo.config.patch(ignore_logger_methods={ignore_method}):
            opt_f = torch.compile(backend="eager")(fn)
            with self.assertLogs(logger, level="INFO") as captured:
                logger.info("call logger info to avoid error")
                opt_out = opt_f(x)
                printed_output = [entry.split(":", 2)[2] for entry in captured.output]

        self.assertTrue(same(orig_out, opt_out))
        if should_ignore_logger:
            self.assertNotIn("moo", printed_output)
            self.assertEqual(len(counters["graph_break"]), 0)
        else:
            self.assertIn("moo", printed_output)
            self.assertEqual(len(counters["graph_break"]), 1)


class ReorderLogsTests(torch._dynamo.test_case.TestCase):
    def test_dont_reorder_print(self):
        def f(x):
            x = x + x
            print("moo")
            x = x * x
            return x

        counters.clear()
        x = torch.randn(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertEqual(printed_output, "moo")
        self.assertEqual(len(counters["graph_break"]), 1)

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_reorder_print(self):
        def f(x):
            print("moo")
            x1 = x + x
            print(x1)
            x2 = x1 * x1
            print(1, 2, 3)
            x3 = x2 + x2
            return (x1, x3)

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(printed_output, f"moo\n{torch.ones(3, 3) * 2}\n1 2 3")
        self.assertTrue(same(orig_out, opt_out))

    @torch._dynamo.config.patch(reorderable_logging_functions={warnings.warn})
    def test_reorder_warnings(self):
        import warnings

        def f(x):
            x1 = x + x
            warnings.warn("moo")
            x2 = x1 * x1
            warnings.warn(f"{x2}")
            x3 = x2 + x2
            return x3

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager", fullgraph=True)(f)
        with warnings.catch_warnings(record=True) as w:
            opt_out = opt_f(x)
            warning_messages = [str(i.message) for i in w]
            orig_out = f(x)

        self.assertTrue(same(orig_out, opt_out))
        self.assertIn("moo", warning_messages)

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_reorder_print_graph_break(self):
        def f(x):
            x1 = x + x
            print(f"res: {x1}")
            x2 = x1 * x1
            torch._dynamo.graph_break()
            x3 = x2 + x2
            print(1, 2, 3)
            return x3

        x = torch.ones(3, 3)
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(x)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(x)

        self.assertEqual(printed_output, f"res: {torch.ones(3, 3) * 2}\n1 2 3")
        self.assertTrue(same(orig_out, opt_out))

    def test_reorder_custom_log_fn(self):
        custom_logs = []

        def custom_log(s: str):
            torch._dynamo.graph_break()
            custom_logs.append(s)

        def f(x):
            custom_log("moo")
            x1 = x + x
            custom_log(f"{x1}")
            return x + x

        x = torch.ones(3, 3)
        counters.clear()
        with torch._dynamo.config.patch(reorderable_logging_functions={custom_log}):
            opt_f = torch.compile(backend="eager")(f)
            opt_f(x)

        self.assertEqual(sum(counters["graph_break"].values()), 1)
        self.assertEqual(custom_logs[0], "moo")
        self.assertEqual(custom_logs[1], f"{torch.ones(3, 3) * 2}")

    @torch._dynamo.config.patch(reorderable_logging_functions={print})
    def test_constant_mutation(self):
        def f(x):
            alist = [x]
            alist.append(x + 1)
            print(alist[-1])
            alist[0].sum().item()  # graph break
            res = alist.pop()
            print(alist[-1])
            res.sum().item()  # graph break
            return res

        inputs = (torch.tensor([1]),)
        counters.clear()
        opt_f = torch.compile(backend="eager")(f)
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            opt_out = opt_f(*inputs)
            printed_output = mock_stdout.getvalue().strip()
            orig_out = f(*inputs)

        self.assertEqual(printed_output, "tensor([2])\ntensor([1])")
        self.assertTrue(same(orig_out, opt_out))

        graph_break_key = counters["graph_break"].keys()
        self.assertEqual(len(graph_break_key), 1)
        self.assertEqual(next(iter(graph_break_key)), "Tensor.item")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
