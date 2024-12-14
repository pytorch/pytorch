# mypy: allow-untyped-defs
# Owner(s): ["module: unknown"]

import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
import unittest
import warnings
from typing import Any, Dict, List

import torch
import torch.cuda
import torch.nn as nn
import torch.utils.cpp_extension
import torch.utils.data
from torch.autograd._functions.utils import check_onnx_broadcast
from torch.onnx.symbolic_opset9 import _prepare_onnx_paddings
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
)
from torch.utils._device import set_device
from torch.utils._pytree import tree_all_only, tree_any
from torch.utils._traceback import (
    CapturedTraceback,
    format_traceback_short,
    report_compile_source_on_error,
)
from torch.utils.checkpoint import (
    _infer_device_type,
    checkpoint,
    checkpoint_sequential,
    get_device_states,
)
from torch.utils.data import DataLoader


# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

HAS_CUDA = torch.cuda.is_available()


from torch.testing._internal.common_utils import run_tests, TestCase


class RandomDatasetMock(torch.utils.data.Dataset):
    def __getitem__(self, index):
        return torch.tensor([torch.rand(1).item(), random.uniform(0, 1)])

    def __len__(self):
        return 1000


class TestCheckpoint(TestCase):
    # This runs checkpoint_sequential on each of the nets in
    # module_lists_to_compare, and compares them against the uncheckpointed model.
    # To compare, it checks outputs as well as input gradients and parameter gradients
    def _check_checkpoint_sequential(
        self,
        model,
        module_lists_to_compare,
        num_chunks,
        input,
        use_reentrant,
    ):
        # not checkpointed
        out = model(input)
        out_not_checkpointed = out.detach().clone()
        model.zero_grad()
        out.sum().backward()
        grad_not_checkpointed = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
        }
        input_grad_not_checkpointed = input.grad.detach().clone()
        for model_to_compare in module_lists_to_compare:
            # checkpointed model by passing list of modules
            detached = input.detach()
            detached.requires_grad = True

            # pass list of modules to checkpoint
            out = checkpoint_sequential(
                model_to_compare, num_chunks, detached, use_reentrant=use_reentrant
            )
            out_checkpointed = out.detach().clone()
            model.zero_grad()
            out.sum().backward()
            grad_checkpointed = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
            }
            input_grad_checkpointed = detached.grad.detach().clone()
            # compare outputs as well as the gradients of input and parameters
            self.assertEqual(out_checkpointed, out_not_checkpointed)
            self.assertEqual(input_grad_not_checkpointed, input_grad_checkpointed)
            for name in grad_checkpointed:
                self.assertEqual(grad_checkpointed[name], grad_not_checkpointed[name])

    # Test whether checkpoint is being triggered or not. For this, we check
    # the number of times forward pass happens
    def test_checkpoint_trigger(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.counter = 0

            def forward(self, input_var):
                self.counter += 1
                # For reentrant, need to have autograd actually
                # pack a tensor to trigger recomp
                ret = input_var * torch.tensor(2.0)
                return ret

        # checkpointed
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                modules = [Net() for _ in range(10)]
                for m in modules:
                    self.assertEqual(m.counter, 0)
                input_var = torch.randn(3, 4, requires_grad=True)
                out = checkpoint_sequential(
                    modules, 2, input_var, use_reentrant=use_reentrant
                )
                for m in modules:
                    self.assertEqual(m.counter, 1)
                out.sum().backward()
                for m in modules[: (len(modules) // 2)]:
                    self.assertEqual(m.counter, 2)
                for m in modules[(len(modules) // 2) :]:
                    self.assertEqual(m.counter, 1)

    def test_checkpoint_valid(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU(),
        )

        input_var = torch.randn(1, 100, requires_grad=True)

        # checkpointed
        chunks = 2
        modules = list(model.children())
        out = checkpoint_sequential(modules, chunks, input_var, use_reentrant=True)
        with self.assertRaisesRegex(
            RuntimeError, "torch.utils.checkpoint is incompatible"
        ):
            torch.autograd.grad(
                outputs=[out],
                grad_outputs=[torch.ones(1, 5)],
                inputs=[input_var],
                create_graph=True,
            )
        # works with use_reentrant=False, and grads are the same
        out = model(input_var)
        grads_no_checkpoint = torch.autograd.grad(
            outputs=[out],
            grad_outputs=[torch.ones(1, 5)],
            inputs=[input_var],
            create_graph=True,
        )
        out_checkpoint = checkpoint_sequential(
            modules, chunks, input_var, use_reentrant=False
        )
        # check outputs are the same
        self.assertEqual(out_checkpoint, out)
        grads_checkpoint = torch.autograd.grad(
            outputs=[out_checkpoint],
            grad_outputs=[torch.ones(1, 5)],
            inputs=[input_var],
            create_graph=True,
        )
        self.assertEqual(grads_no_checkpoint, grads_checkpoint)

    def test_checkpoint(self):
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                model = nn.Sequential(
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                    nn.ReLU(),
                )

                # Compare uncheckpointed model with its checkpointed counterparts
                # In addition to running checkpoint_sequential on the nn.Sequential
                # instance, we also run the function on the list of functions within
                # the module.
                self._check_checkpoint_sequential(
                    model,
                    [list(model.children()), model],
                    2,
                    torch.randn(1, 100, requires_grad=True),
                    use_reentrant=use_reentrant,
                )

    def test_checkpoint_module_list(self):
        class ModuleListNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                module_list = [
                    nn.Linear(100, 50),
                    nn.ReLU(),
                    nn.Linear(50, 20),
                    nn.ReLU(),
                    nn.Linear(20, 5),
                    nn.ReLU(),
                ]
                self.module_list = nn.ModuleList(module_list)

            def forward(self, input):
                for layer in self.module_list:
                    input = layer(input)
                return input

        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                model = ModuleListNet()

                # Compare uncheckpointed model with its checkpointed counterparts.
                self._check_checkpoint_sequential(
                    model,
                    [list(model.module_list.children()), model.module_list],
                    2,
                    torch.randn(1, 100, requires_grad=True),
                    use_reentrant=use_reentrant,
                )

    def test_checkpoint_sequential_deprecated_multiple_args(self):
        class Two(nn.Module):
            def forward(self, a, b):
                return a, b

        model = nn.Sequential(Two())
        a = torch.randn(1, 100, requires_grad=True)
        b = torch.randn(1, 100, requires_grad=True)

        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                with self.assertRaises(TypeError):
                    checkpoint_sequential(model, 1, a, b)

    def test_checkpoint_sequential_deprecated_no_args(self):
        class Noop(nn.Module):
            def forward(self):
                pass

        model = nn.Sequential(Noop())
        for use_reentrant in [True, False]:
            with self.subTest(use_reentrant=use_reentrant):
                with self.assertRaises(TypeError):
                    checkpoint_sequential(model, 1)  # type: ignore[call-arg]

    def test_checkpoint_rng_cpu(self):
        for _ in range(5):
            inp = torch.randn(20000, device="cpu").requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                return phase2(input)

            state = torch.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out, use_reentrant=True)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_checkpoint_rng_cuda(self):
        for _ in range(5):
            inp = torch.randn(20000, device="cuda").requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                return phase2(input)

            state = torch.cuda.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out, use_reentrant=True)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.cuda.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_checkpoint_not_preserve_rng_state_and_without_reentrant(self):
        inp = torch.randn(2, device="cuda").requires_grad_()
        layer = torch.nn.Dropout()

        def run_fn(input):
            return layer(input)

        out = checkpoint(run_fn, inp, use_reentrant=False, preserve_rng_state=False)
        out.sum().backward()
        # This should run without error

    def test_checkpoint_non_tensor(self):
        def run_fn(tensor1, tensor2):
            if tensor2 is None:
                return tensor1
            return tensor1 + tensor2

        input_var = torch.randn(1, 100, requires_grad=True)
        out = checkpoint(run_fn, input_var, None, use_reentrant=True)
        out.sum().backward()

    def test_checkpoint_non_tensor_inputs_outputs(self):
        def foo(t1, t2, scale, t3):
            t4 = t1 + t2 * t3
            t5 = t1 * t2 + t3
            t4 *= scale
            t5 *= scale
            return scale, t4, None, True, t5, "bar", t1

        t1 = torch.rand(10, requires_grad=True)
        t2 = torch.rand(10, requires_grad=True)
        t3 = torch.rand(10)
        scale = random.randint(0, 10)
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

        # Validate running backward.
        res[1].sum().backward(retain_graph=True)
        res[4].sum().backward(retain_graph=True)
        res[6].sum().backward()
        with self.assertRaisesRegex(
            RuntimeError, "Trying to backward through the graph a second time"
        ):
            res[6].sum().backward()
        t1_grad = t1.grad
        t2_grad = t2.grad

        # Reset grads, run without checkpoint and validate we receive same grads.
        t1.grad = None
        t2.grad = None
        res = foo(t1, t2, scale, t3)
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertEqual(t1.grad, t1_grad)
        self.assertEqual(t2.grad, t2_grad)

    def test_checkpoint_no_tensors(self):
        def foo(t1, t2, scale, t3):
            t4 = t1 + t2 * t3
            t5 = t1 * t2 + t3
            t4 *= scale
            t5 *= scale
            return scale, t4, None, True, t5, "bar", t1

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = checkpoint(foo, t1, t2, scale, t3, use_reentrant=True)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_checkpoint_partial_grad(self):
        def run_fn(tensor1, tensor2):
            # tensor 2 is used for other application logic
            return tensor1, tensor2

        input_var = torch.randn(1, 4, requires_grad=True)
        input_var2 = torch.randn(1, 4, requires_grad=False)
        out = checkpoint(run_fn, input_var, input_var2, use_reentrant=True)
        out[0].sum().backward()

        def run_fn2(tensor1, tensor2):
            return tensor1

        input_var = torch.randn(1, 4, requires_grad=False)
        input_var2 = torch.randn(1, 4, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"none of output has requires_grad=True, this checkpoint\(\) is not necessary",
        ):
            out = checkpoint(run_fn2, input_var, input_var2, use_reentrant=True)
            out.sum().backward()

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpointing_without_reentrant_early_free(self):
        # I don't know how to check if the temporary saved variable buffer
        # get de-allocated directly. So using cuda memory usage as a proxy

        def _do_test(fn, should_free):
            stats: List[int] = []

            def track(x, idx):
                # Track that at each step of the backward, some Tensor were
                # de-allocated (which correspond to the checkpoint storage being
                # emptied at each step)
                def hook(_unused):
                    self.assertEqual(len(stats), idx)
                    torch.cuda.synchronize()
                    stats.append(torch.cuda.memory_allocated())
                    if idx > 0:
                        if should_free:
                            self.assertLess(stats[idx], stats[idx - 1])
                        else:
                            self.assertEqual(stats[idx], stats[idx - 1])

                x.register_hook(hook)

            def test_fn(x):
                # The main property of this function is that it contains multiple
                # operations that save gradients in a chain.
                x = x**2
                track(x, 2)
                x = x**2
                track(x, 1)
                x = x**2
                track(x, 0)
                x = x**2
                return x.sum()

            fn(test_fn)

            return stats

        x = torch.zeros(10, device="cuda", requires_grad=True)
        x.grad = torch.zeros_like(x)

        # In a regular backward, buffers get eagerly freed
        non_retain_stats = _do_test(lambda fn: fn(x).backward(), True)

        # In a retain_grad backward, buffers get preserved
        _unused_retain_stats = _do_test(
            lambda fn: fn(x).backward(retain_graph=True), False
        )

        # In a regular backward with checkpoint, buffers get eagerly freed
        checkpoint_non_retain_stats = _do_test(
            lambda fn: checkpoint(fn, x, use_reentrant=False).backward(), True
        )

        # In a retain_grad backward with checkpoint, buffers get eagerly freed
        checkpoint_retain_stats = _do_test(
            lambda fn: checkpoint(fn, x, use_reentrant=False).backward(
                retain_graph=True
            ),
            True,
        )

        self.assertEqual(non_retain_stats, checkpoint_non_retain_stats)
        self.assertEqual(non_retain_stats, checkpoint_retain_stats)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_get_device_states_recursive(self):
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="cuda:1")],
        }
        device_ids, device_states = get_device_states(inp)
        self.assertEqual(2, len(device_ids))
        self.assertEqual(2, len(device_states))
        self.assertEqual(0, device_ids[0])
        self.assertEqual(1, device_ids[1])
        self.assertTrue(isinstance(device_states[0], torch.Tensor))
        self.assertTrue(isinstance(device_states[1], torch.Tensor))

    def test_infer_device_state_recursive_meta(self):
        inp = {"foo": torch.rand(10, device="meta")}
        device_type = _infer_device_type(inp)
        self.assertEqual("meta", device_type)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_infer_device_state_recursive_multi_cuda(self):
        # Check that no warning is issued for either cuda:0, cuda:1 or
        # cuda:0, cuda:0 cases since they are both the same device type
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="cuda:1")],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device_type = _infer_device_type(inp)
            self.assertEqual("cuda", device_type)
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="cuda:0")],
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            device_type = _infer_device_type(inp)
            self.assertEqual("cuda", device_type)
        # Check that a warning is issued for cuda:0, meta and that it includes
        # device type information
        inp = {
            "foo": torch.rand(10, device="cuda:0"),
            "bar": [torch.rand(10, device="meta")],
        }
        with warnings.catch_warnings(record=True) as w:
            device_type = _infer_device_type(inp)
            self.assertEqual("cuda", device_type)
        self.assertEqual(len(w), 1)
        warning_msg = str(w[-1].message)
        self.assertTrue(
            "Tensor arguments, excluding CPU tensors, are detected on at least two types of devices"
            in warning_msg
        )
        self.assertTrue("Device types: ['cuda', 'meta']" in warning_msg)
        self.assertTrue("first device type: cuda" in warning_msg)


class TestDataLoaderUtils(TestCase):
    MAX_TIMEOUT_IN_SECOND = 300

    def setUp(self):
        super().setUp()
        self.dataset = torch.randn(5, 3, 3, 2)
        self.batch_size = 3

    def test_random_seed(self):
        def run():
            dataloader = torch.utils.data.DataLoader(
                RandomDatasetMock(),
                batch_size=2,
                num_workers=4,
                shuffle=True,
                timeout=self.MAX_TIMEOUT_IN_SECOND,
            )
            return next(iter(dataloader))

        torch.manual_seed(2018)
        x1 = run()
        torch.manual_seed(2018)
        x2 = run()
        self.assertEqual(x1, x2)

    def test_single_keep(self):
        # self.dataset is a Tensor here; technically not a valid input because
        # not a Dataset subclass, but needs to stay working so add ignore's
        # for type checking with mypy
        dataloader: DataLoader = DataLoader(
            self.dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            num_workers=0,
            drop_last=False,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 2)

    def test_single_drop(self):
        dataloader: DataLoader = DataLoader(
            self.dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            num_workers=0,
            drop_last=True,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 1)

    @unittest.skip(
        "FIXME: Intermittent CUDA out-of-memory error on Windows and time-out under ASAN"
    )
    def test_multi_keep(self):
        dataloader: DataLoader = DataLoader(
            self.dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            num_workers=2,
            drop_last=False,
            timeout=self.MAX_TIMEOUT_IN_SECOND,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 2)

    def test_multi_drop(self):
        dataloader: DataLoader = DataLoader(
            self.dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            num_workers=2,
            drop_last=True,
            timeout=self.MAX_TIMEOUT_IN_SECOND,
        )
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 1)


test_dir = os.path.abspath(os.path.dirname(str(__file__)))


@unittest.skipIf(
    "SKIP_TEST_BOTTLENECK" in os.environ.keys(), "SKIP_TEST_BOTTLENECK is set"
)
class TestBottleneck(TestCase):
    def _run(self, command, timeout=30):
        """Returns (return-code, stdout, stderr)"""
        import subprocess

        p = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
        )
        try:
            output, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            output, err = p.communicate()
        rc = p.returncode
        output_str = output.decode("ascii")
        err_str = err.decode("ascii")
        return (rc, output_str, err_str)

    def _run_bottleneck(self, test_file, scriptargs=""):
        curdir = os.path.dirname(os.path.abspath(__file__))
        filepath = f"{curdir}/{test_file}"
        if scriptargs != "":
            scriptargs = f" {scriptargs}"
        rc, out, err = self._run(
            f"{sys.executable} -m torch.utils.bottleneck {filepath}{scriptargs}"
        )
        return rc, out, err

    def _check_run_args(self):
        # Check that this fails due to missing args
        rc, out, err = self._run_bottleneck("bottleneck_test/test_args.py")
        self.assertEqual(
            rc,
            2,
            atol=0,
            rtol=0,
            msg=self._fail_msg("Missing args should error", out + err),
        )

        # This should succeed
        rc, out, err = self._run_bottleneck(
            "bottleneck_test/test_args.py", "--foo foo --bar bar"
        )
        self.assertEqual(
            rc,
            0,
            atol=0,
            rtol=0,
            msg=self._fail_msg("Should pass args to script", out + err),
        )

    def _fail_msg(self, msg, output):
        return f"{msg}, output was:\n{output}"

    def _check_environment_summary(self, output):
        results = re.search("Environment Summary", output)
        self.assertIsNotNone(
            results, self._fail_msg("Should have Environment Summary", output)
        )

        # Up to five lines away from the heading, there should be the version number
        results = re.search(
            r"Environment Summary.*(\n.*){,5}\nPyTorch \d+\.\d+", output
        )
        self.assertIsNotNone(
            results, self._fail_msg("Should have PyTorch version", output)
        )

    def _check_cprof_summary(self, output):
        results = re.search("cProfile output", output)
        self.assertIsNotNone(
            results, self._fail_msg("Should have cProfile output", output)
        )

        # This assumes that after the cProfile output section we have
        # the autograd profiler output
        results = re.search(
            r"cProfile output.*(\n.*){6,50}\n.*autograd profiler output", output
        )
        self.assertIsNotNone(
            results,
            self._fail_msg(
                "Distance between cProfile and autograd prof out not in [6, 50] lines",
                output,
            ),
        )

    def _check_autograd_summary(self, output):
        results = re.search("autograd profiler output", output)
        self.assertIsNotNone(
            results, self._fail_msg("Should have autograd profiler output", output)
        )

        # This assumes that after the autograd profiler output is the end of the
        # output.
        results = re.search(r"autograd profiler output.*(\n.*){6,100}", output)
        self.assertIsNotNone(
            results,
            self._fail_msg(
                "Distance between autograd prof output and end of output not in [6, 100] lines",
                output,
            ),
        )

    def _check_cuda(self, output):
        if HAS_CUDA:
            results = re.search("CUDA mode", output)
            self.assertIsNotNone(
                results, self._fail_msg("Should tell users CUDA", output)
            )
        else:
            results = re.search("CUDA mode", output)
            self.assertIsNone(
                results, self._fail_msg("Should not tell users about CUDA", output)
            )

    @unittest.skipIf(HAS_CUDA, "CPU-only test")
    def test_bottleneck_cpu_only(self):
        rc, out, err = self._run_bottleneck("bottleneck_test/test.py")
        self.assertEqual(rc, 0, msg=f"Run failed with\n{err}")

        self._check_run_args()
        self._check_environment_summary(out)
        self._check_autograd_summary(out)
        self._check_cprof_summary(out)
        self._check_cuda(out)

    @unittest.skipIf(not HAS_CUDA, "No CUDA")
    def test_bottleneck_cuda(self):
        rc, out, err = self._run_bottleneck("bottleneck_test/test_cuda.py")
        self.assertEqual(rc, 0, msg=f"Run failed with\n{err}")

        self._check_run_args()
        self._check_environment_summary(out)
        self._check_autograd_summary(out)
        self._check_cprof_summary(out)
        self._check_cuda(out)


from torch.utils.collect_env import get_pretty_env_info


@unittest.skipIf(IS_FBCODE, "runs pip which is not available internally")
class TestCollectEnv(TestCase):
    def test_smoke(self):
        info_output = get_pretty_env_info()
        self.assertTrue(info_output.count("\n") >= 17)


class TestONNXUtils(TestCase):
    def test_prepare_onnx_paddings(self):
        sizes = [2, 3, 4]
        pad = [1, 2, 3, 4]
        paddings = _prepare_onnx_paddings(len(sizes), pad)
        self.assertEqual(paddings, [0, 3, 1, 0, 4, 2])

    def test_check_onnx_broadcast(self):
        def try_check_onnx_broadcast(dims1, dims2, expect_broadcast, expect_fail):
            broadcast = True
            fail = False
            try:
                broadcast = check_onnx_broadcast(dims1, dims2)
            except ValueError:
                fail = True
            self.assertEqual(broadcast, expect_broadcast)
            self.assertEqual(fail, expect_fail)

        # Case 1, check the case when len(dims1) < len(dims2) and numel(dims2) > 1
        dims1 = [3, 4]
        dims2 = [2, 3, 4]
        try_check_onnx_broadcast(dims1, dims2, True, True)

        # Case 2, check the case when len(dims1) < len(dims2) and numel(dims2) == 1
        dims1 = [3, 4]
        dims2 = [1, 1, 1]
        try_check_onnx_broadcast(dims1, dims2, True, False)

        # Case 3, check the case when len(dims1) > len(dims2) and numel(dims2) == 1
        dims1 = [1, 1]
        dims2 = [1]
        try_check_onnx_broadcast(dims1, dims2, True, False)

        # Case 4, check the case when len(dims1) > len(dims2) and dims1[x:] == dims2
        dims1 = [2, 3, 4]
        dims2 = [3, 4]
        try_check_onnx_broadcast(dims1, dims2, True, False)

        # Case 5, check the case when len(dims1) > len(dims2), but dims1[x:] != dims2
        dims1 = [2, 3, 4]
        dims2 = [1, 4]
        try_check_onnx_broadcast(dims1, dims2, True, True)

        # Case 6, check the equal case, no broadcast
        dims1 = [3, 4]
        dims2 = [3, 4]
        try_check_onnx_broadcast(dims1, dims2, False, False)

        # Case 7, check the case when len(dims1) == len(dims2), but dims1 != dims2
        dims1 = [3, 4]
        dims2 = [1, 4]
        try_check_onnx_broadcast(dims1, dims2, True, True)

        # Case 8, check the case when len(dims1) == len(dims2) and numel(s2) == 1
        dims1 = [3, 4]
        dims2 = [1, 1]
        try_check_onnx_broadcast(dims1, dims2, True, False)


class TestHipify(TestCase):
    def test_import_hipify(self):
        from torch.utils.hipify import hipify_python  # noqa: F401


class TestHipifyTrie(TestCase):
    def setUp(self):
        self.trie = torch.utils.hipify.hipify_python.Trie()

    def test_add_and_search_trie(self):
        self.trie.add("banana")
        self.assertTrue(self.trie.search("banana"))
        self.assertFalse(self.trie.search("ban"))
        self.assertFalse(self.trie.search("dog"))

    def test_add_multiple_and_search_trie(self):
        words_to_add = ["banana", "apple", "orange"]
        for word in words_to_add:
            self.trie.add(word)

        for word in words_to_add:
            self.assertTrue(self.trie.search(word))

        for word in ["ban", "dog", "okay", "app"]:
            self.assertFalse(self.trie.search(word))

    def test_quote_escape(self):
        orig_chars = ["*", "[", ".", "+", "a", "z", "-"]
        quoted_strs = ["\\*", "\\[", "\\.", "\\+", "a", "z", "\\-"]
        for i in range(len(orig_chars)):
            self.assertEqual(self.trie.quote(orig_chars[i]), quoted_strs[i])

    def test_export_trie_to_regex(self):
        words_to_add = [
            "__CUDACC__",
            "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
            "CUDA_ERROR_ARRAY_IS_MAPPED",
            "CUDA_ERROR_NOT_MAPPED",
            "CUDA_ERROR_INVALID_SOURCE",
        ]
        for word in words_to_add:
            self.trie.add(word)
        regex = self.trie.export_to_regex()
        expected_regex = r"(?:CUDA_ERROR_(?:ARRAY_IS_MAPPED|CONTEXT_ALREADY_CURRENT|INVALID_SOURCE|NOT_MAPPED)|__CUDACC__)"
        self.assertEqual(regex, expected_regex)

    def test_prefix_words_export_trie_to_regex(self):
        # test case where some nodes have both children and are also leaf nodes.
        words_to_add = ["apple", "app", "ban", "banana"]
        for word in words_to_add:
            self.trie.add(word)
        regex = self.trie.export_to_regex()
        expected_regex = r"(?:app(?:le)?|ban(?:ana)?)"
        self.assertEqual(regex, expected_regex)

    def test_single_export_trie_to_regex(self):
        words_to_add = ["cudaErrorInvalidMemcpyDirection"]
        for word in words_to_add:
            self.trie.add(word)
        regex = self.trie.export_to_regex()
        expected_regex = "cudaErrorInvalidMemcpyDirection"
        self.assertEqual(regex, expected_regex)

    def test_char_export_trie_to_regex(self):
        self.trie.add("a")
        self.assertEqual(self.trie.export_to_regex(), "a")
        self.trie.add("b")
        self.assertEqual(self.trie.export_to_regex(), "[ab]")

    def test_special_char_export_trie_to_regex(self):
        self.trie.add(r"c*")
        self.assertEqual(self.trie.export_to_regex(), r"c\*")


class TestAssert(TestCase):
    def test_assert_true(self):
        # verify assertions work as expected
        # bool argument
        torch._assert(True, "foo")
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(False, "bar")
        # tensor argument
        torch._assert(torch.tensor([True], dtype=torch.bool), "foo")
        with self.assertRaisesRegex(AssertionError, "bar"):
            torch._assert(torch.tensor([False], dtype=torch.bool), "bar")

    def test_assert_scriptable(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch._assert(x.sum() > 0, "foo")
                return x

        m = M()
        # scriptable
        ms = torch.jit.script(m)
        # data can be passed without errors
        x = torch.randn(4, 4).fill_(1.0)
        ms(x)
        with self.assertRaisesRegex(torch.jit.Error, "foo"):
            ms(torch.tensor([False], dtype=torch.bool))


@unittest.skipIf(IS_SANDCASTLE, "cpp_extension is OSS only")
class TestStandaloneCPPJIT(TestCase):
    def test_load_standalone(self):
        build_dir = tempfile.mkdtemp()
        try:
            src_path = os.path.join(build_dir, "main.cpp")
            src = textwrap.dedent(
                """\
                #include <iostream>
                #include <torch/torch.h>
                int main() {
                    auto x = torch::eye(3);
                    std::cout << x << std::endl;
                }
            """
            )
            with open(src_path, "w") as f:
                f.write(src)

            exec_path = torch.utils.cpp_extension.load(
                "standalone_load_test",
                src_path,
                build_directory=build_dir,
                is_python_module=False,
                is_standalone=True,
            )

            ext = ".exe" if IS_WINDOWS else ""
            self.assertEqual(
                exec_path, os.path.join(build_dir, f"standalone_load_test{ext}")
            )

            for shell in [True, False]:
                r = subprocess.run(
                    [exec_path],
                    shell=shell,
                    stdout=subprocess.PIPE,
                )
                self.assertEqual(r.returncode, 0)
                self.assertEqual(
                    # Windows prints "\r\n" for newlines.
                    textwrap.dedent(r.stdout.decode("utf-8")).replace("\r\n", "\n"),
                    textwrap.dedent(
                        """\
                     1  0  0
                     0  1  0
                     0  0  1
                    [ CPUFloatType{3,3} ]
                    """
                    ),
                )

        finally:
            shutil.rmtree(build_dir)


class DummyPrivateUse1Module:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_autocast_enabled():
        return True

    @staticmethod
    def get_autocast_dtype():
        return torch.float16

    @staticmethod
    def set_autocast_enabled(enable):
        pass

    @staticmethod
    def set_autocast_dtype(dtype):
        pass

    @staticmethod
    def get_amp_supported_dtype():
        return [torch.float16]


class TestExtensionUtils(TestCase):
    def tearDown(self):
        # Clean up
        backend_name = torch._C._get_privateuse1_backend_name()
        if hasattr(torch, backend_name):
            delattr(torch, backend_name)
        if f"torch.{backend_name}" in sys.modules:
            del sys.modules[f"torch.{backend_name}"]

    def test_external_module_register(self):
        # Built-in module
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("cuda", torch.cuda)

        # Wrong device type
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dummmy", DummyPrivateUse1Module)

        with self.assertRaises(AttributeError):
            torch.privateuseone.is_available()  # type: ignore[attr-defined]

        torch._register_device_module("privateuseone", DummyPrivateUse1Module)

        torch.privateuseone.is_available()  # type: ignore[attr-defined]

        # No supporting for override
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("privateuseone", DummyPrivateUse1Module)

    def test_external_module_register_with_renamed_backend(self):
        torch.utils.rename_privateuse1_backend("foo")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dummmy")

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(custom_backend_name, "foo")

        with self.assertRaises(AttributeError):
            torch.foo.is_available()  # type: ignore[attr-defined]

        with self.assertRaisesRegex(AssertionError, "Tried to use AMP with the"):
            with torch.autocast(device_type=custom_backend_name):
                pass
        torch._register_device_module("foo", DummyPrivateUse1Module)

        torch.foo.is_available()  # type: ignore[attr-defined]
        with torch.autocast(device_type=custom_backend_name):
            pass

        self.assertEqual(torch._utils._get_device_index("foo:1"), 1)
        self.assertEqual(torch._utils._get_device_index(torch.device("foo:2")), 2)


class TestRenderUtils(TestCase):
    def test_basic(self):
        self.assertExpectedInline(
            torch._utils.render_call(torch.sum, [torch.randn(100)], {"dim": 0}),
            """torch.sum(tensor([...], size=(100,)), dim=0)""",
        )
        self.assertExpectedInline(
            torch._utils.render_call(torch.sum, [torch.randn(100, 100)], {"dim": 0}),
            """torch.sum(tensor([...], size=(100, 100)), dim=0)""",
        )


class TestDeviceUtils(TestCase):
    def test_basic(self):
        with torch.device("meta") as dev:
            x = torch.empty(3, 3)
        self.assertEqual(x.device.type, "meta")
        self.assertEqual(dev, torch.device("meta"))

    def test_decorator(self):
        @set_device("meta")
        def f():
            return torch.empty(3, 3)

        self.assertEqual(f().device.type, "meta")

    def test_decorator_generator(self):
        @set_device("meta")
        def f():
            yield torch.empty(3, 3)
            yield torch.empty(3, 3)

        r1, r2 = list(f())
        self.assertEqual(r1.device.type, "meta")
        self.assertEqual(r2.device.type, "meta")

    def test_nn_module(self):
        with torch.device("meta"):
            m = nn.Linear(40, 50)
        self.assertEqual(m.weight.device.type, "meta")

    def test_set_default_device(self):
        try:
            torch.set_default_device("meta")
            r = torch.empty(2, 2)
        finally:
            torch.set_default_device(None)

        self.assertEqual(r.device.type, "meta")

    def test_get_default_device(self):
        torch.set_default_device("meta")
        self.assertEqual(torch.get_default_device().type, "meta")
        torch.set_default_device(None)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_get_default_device_more(self):
        torch.set_default_device("cuda")
        self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
        torch.set_default_device(None)

        torch.set_default_device("cuda")
        torch.cuda.set_device("cuda:1")
        self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
        torch.set_default_device(None)

        torch.set_default_device("cuda:1")
        self.assertEqual(torch.get_default_device(), torch.tensor([]).device)
        torch.set_default_device(None)

    @onlyCPU
    @ops(op_db)
    def test_device_mode_ops(self, device, dtype, op):
        func = op.get_op()
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            # Only test samples which don't have Tensor inputs.  However,
            # we don't test the factory property on OpInfo as it is very,
            # very incomplete
            if tree_any(
                lambda x: isinstance(x, torch.Tensor),
                (sample.input, sample.args, sample.kwargs),
            ):
                continue
            # Many OpInfos will explicitly pass in a device.  DeviceContext
            # will respect device if it is explicitly specified.  To test
            # DeviceContext, we have to remove the device kwarg in this case.
            # NB: Can't pass None to sample_inputs, the function can't
            # handle it.
            kwargs = sample.kwargs.copy()
            kwargs.pop("device", None)
            with torch.device("meta"):
                r = func(sample.input, *sample.args, **kwargs)

            def is_meta_device(x: torch.Tensor) -> bool:
                return x.device.type == "meta"

            self.assertTrue(tree_all_only(torch.Tensor, is_meta_device, r))


instantiate_device_type_tests(TestDeviceUtils, globals())


class TestCppExtensionUtils(TestCase):
    def test_cpp_compiler_is_ok(self):
        self.assertTrue(torch.utils.cpp_extension.check_compiler_ok_for_platform("c++"))

    def test_cc_compiler_is_ok(self):
        self.assertTrue(torch.utils.cpp_extension.check_compiler_ok_for_platform("cc"))


class TestTraceback(TestCase):
    def test_basic(self):
        source = """\
def f(x):
    def g(x):
        raise RuntimeError  # HEYA

    x = x * 3
    return g(x) + 1
"""

        out: Dict[str, Any] = {}
        scope = {"__compile_source__": source}
        exec(source, scope, out)

        try:
            with report_compile_source_on_error():
                out["f"](1)
        except RuntimeError as e:
            self.assertIn("HEYA", "".join(traceback.format_tb(e.__traceback__)))

    def test_format_traceback_short(self):
        try:
            raise RuntimeError
        except RuntimeError as e:
            self.assertRegex(
                format_traceback_short(e.__traceback__),
                r".*test_utils.py:\d+ in test_format_traceback_short",
            )

    def test_captured_traceback(self):
        self.assertIn(
            "test_captured_traceback", "".join(CapturedTraceback.extract().format())
        )

    def test_captured_traceback_format_all(self):
        rs = CapturedTraceback.format_all(
            [CapturedTraceback.extract(), CapturedTraceback.extract()]
        )
        self.assertEqual(len(rs), 2)
        self.assertIn("test_captured_traceback_format_all", "".join(rs[0]))

    def test_captured_traceback_format_all_cached(self):
        tb = CapturedTraceback.extract()
        tb.format()  # cached
        rs = CapturedTraceback.format_all([tb, CapturedTraceback.extract()])
        self.assertEqual(len(rs), 2)
        self.assertIn("test_captured_traceback_format_all", "".join(rs[0]))


if __name__ == "__main__":
    run_tests()
