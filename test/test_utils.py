from __future__ import print_function
import sys
import os
import re
import shutil
import random
import tempfile
import unittest
import torch
import torch.nn as nn
import torch.utils.data
import torch.cuda
from torch._six import PY2
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.hub as hub
from torch.autograd._functions.utils import check_onnx_broadcast
from torch.onnx.symbolic_opset9 import _prepare_onnx_paddings
from torch.testing._internal.common_utils import skipIfRocm, load_tests, retry, IS_SANDCASTLE
if PY2:
    from urllib2 import HTTPError
else:
    from urllib.error import HTTPError

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

HAS_CUDA = torch.cuda.is_available()

from torch.testing._internal.common_utils import TestCase, run_tests


class RandomDatasetMock(object):

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
            out = checkpoint_sequential(model_to_compare, num_chunks, detached)
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

            def __init__(self):
                super(Net, self).__init__()
                self.counter = 0

            def forward(self, input_var):
                self.counter += 1
                return input_var

        # checkpointed
        modules = [Net() for _ in range(10)]
        for m in modules:
            self.assertEqual(m.counter, 0)
        input_var = torch.randn(3, 4, requires_grad=True)
        out = checkpoint_sequential(modules, 2, input_var)
        for m in modules:
            self.assertEqual(m.counter, 1)
        out.sum().backward()
        for m in modules[:(len(modules) // 2)]:
            self.assertEqual(m.counter, 2)
        for m in modules[(len(modules) // 2):]:
            self.assertEqual(m.counter, 1)

    def test_checkpoint_valid(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        input_var = torch.randn(1, 100, requires_grad=True)

        # checkpointed
        chunks = 2
        modules = list(model.children())
        out = checkpoint_sequential(modules, chunks, input_var)
        # python_error in case of py2_7_9.
        with self.assertRaisesRegex(RuntimeError, "(Checkpointing is not compatible)|(python_error)"):
            torch.autograd.grad(
                outputs=[out], grad_outputs=[torch.ones(1, 5)], inputs=[input_var], create_graph=True
            )

    def test_checkpoint(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        # Compare uncheckpointed model with its checkpointed counterparts
        # In addition to running checkpoint_sequential on the nn.Sequential
        # instance, we also run the function on the list of functions within
        # the module.
        self._check_checkpoint_sequential(
            model,
            [list(model.children()), model],
            2,
            torch.randn(1, 100, requires_grad=True)
        )

    def test_checkpoint_module_list(self):
        class ModuleListNet(nn.Module):
            def __init__(self):
                super(ModuleListNet, self).__init__()
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

        model = ModuleListNet()

        # Compare uncheckpointed model with its checkpointed counterparts.
        self._check_checkpoint_sequential(
            model,
            [list(model.module_list.children()), model.module_list],
            2,
            torch.randn(1, 100, requires_grad=True),
        )

    def test_checkpoint_sequential_deprecated_multiple_args(self):
        class Two(nn.Module):
            def forward(self, a, b):
                return a, b

        model = nn.Sequential(Two())
        a = torch.randn(1, 100, requires_grad=True)
        b = torch.randn(1, 100, requires_grad=True)

        with self.assertRaises(TypeError):
            checkpoint_sequential(model, 1, a, b)

    def test_checkpoint_sequential_deprecated_no_args(self):
        class Noop(nn.Module):
            def forward(self):
                pass

        model = nn.Sequential(Noop())

        with self.assertRaises(TypeError):
            checkpoint_sequential(model, 1)

    def test_checkpoint_rng_cpu(self):
        for _ in range(5):
            inp = torch.randn(20000, device='cpu').requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                return phase2(input)

            state = torch.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    @unittest.skipIf(not HAS_CUDA, 'No CUDA')
    def test_checkpoint_rng_cuda(self):
        for _ in range(5):
            inp = torch.randn(20000, device='cuda').requires_grad_()
            phase1 = torch.nn.Dropout()
            phase2 = torch.nn.Dropout()

            def run_fn(input):
                return phase2(input)

            state = torch.cuda.get_rng_state()

            out = phase1(inp)
            out = checkpoint(run_fn, out)
            out.sum().backward()
            grad_with_checkpointing = inp.grad

            torch.cuda.set_rng_state(state)

            inp.grad = None

            out = phase1(inp)
            out = run_fn(out)
            out.sum().backward()
            grad_no_checkpointing = inp.grad

            self.assertEqual(grad_with_checkpointing, grad_no_checkpointing)

    def test_checkpoint_non_tensor(self):

        def run_fn(tensor1, tensor2):
            if tensor2 is None:
                return tensor1
            return tensor1 + tensor2

        input_var = torch.randn(1, 100, requires_grad=True)
        out = checkpoint(run_fn, input_var, None)
        out.sum().backward()


class TestDataLoader(TestCase):
    def setUp(self):
        self.dataset = torch.randn(5, 3, 3, 2)
        self.batch_size = 3

    def test_random_seed(self):
        def run():
            dataloader = torch.utils.data.DataLoader(RandomDatasetMock(),
                                                     batch_size=2,
                                                     num_workers=4,
                                                     shuffle=True)
            return next(iter(dataloader))

        torch.manual_seed(2018)
        x1 = run()
        torch.manual_seed(2018)
        x2 = run()
        self.assertEqual(x1, x2)

    def test_single_keep(self):
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=0,
                                                 drop_last=False)
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 2)

    def test_single_drop(self):
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=0,
                                                 drop_last=True)
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 1)

    @unittest.skip("FIXME: Intermittent CUDA out-of-memory error on Windows and time-out under ASAN")
    def test_multi_keep(self):
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=2,
                                                 drop_last=False)
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 2)

    def test_multi_drop(self):
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size=self.batch_size,
                                                 num_workers=2,
                                                 drop_last=True)
        dataiter = iter(dataloader)
        self.assertEqual(len(list(dataiter)), 1)


test_dir = os.path.abspath(os.path.dirname(str(__file__)))


class TestFFI(TestCase):
    def test_deprecated(self):
        with self.assertRaisesRegex(ImportError, "torch.utils.ffi is deprecated. Please use cpp extensions instead."):
            from torch.utils.ffi import create_extension  # noqa: F401


@unittest.skipIf('SKIP_TEST_BOTTLENECK' in os.environ.keys(), 'SKIP_TEST_BOTTLENECK is set')
class TestBottleneck(TestCase):
    def _run(self, command):
        """Returns (return-code, stdout, stderr)"""
        import subprocess
        from torch.testing._internal.common_utils import PY3

        p = subprocess.Popen(command, stdout=subprocess.PIPE,  # noqa
                             stderr=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        rc = p.returncode
        if PY3:
            output = output.decode("ascii")
            err = err.decode("ascii")
        return (rc, output, err)

    def _run_bottleneck(self, test_file, scriptargs=''):
        curdir = os.path.dirname(os.path.abspath(__file__))
        filepath = '{}/{}'.format(curdir, test_file)
        if scriptargs != '':
            scriptargs = ' {}'.format(scriptargs)
        rc, out, err = self._run(
            '{} -m torch.utils.bottleneck {}{}'.format(sys.executable, filepath, scriptargs))
        return rc, out, err

    def _check_run_args(self):
        # Check that this fails due to missing args
        rc, out, err = self._run_bottleneck('bottleneck_test/test_args.py')
        self.assertEqual(rc, 2, None, self._fail_msg('Missing args should error', out + err))

        # This should succeed
        rc, out, err = self._run_bottleneck('bottleneck_test/test_args.py', '--foo foo --bar bar')
        self.assertEqual(rc, 0, None, self._fail_msg('Should pass args to script', out + err))

    def _fail_msg(self, msg, output):
        return '{}, output was:\n{}'.format(msg, output)

    def _check_environment_summary(self, output):
        results = re.search('Environment Summary', output)
        self.assertIsNotNone(results, self._fail_msg('Should have Environment Summary', output))

        # Up to five lines away from the heading, there should be the version number
        results = re.search(r'Environment Summary.*(\n.*){,5}\nPyTorch \d+\.\d+', output)
        self.assertIsNotNone(results, self._fail_msg('Should have PyTorch version', output))

    def _check_cprof_summary(self, output):
        results = re.search('cProfile output', output)
        self.assertIsNotNone(results, self._fail_msg('Should have cProfile output', output))

        # This assumes that after the cProfile output section we have
        # the autograd profiler output
        results = re.search(r'cProfile output.*(\n.*){6,50}\n.*autograd profiler output', output)
        self.assertIsNotNone(results, self._fail_msg(
            'Distance between cProfile and autograd prof out not in [6, 50] lines', output))

    def _check_autograd_summary(self, output):
        results = re.search('autograd profiler output', output)
        self.assertIsNotNone(results, self._fail_msg('Should have autograd profiler output', output))

        # This assumes that after the autograd profiler output is the end of the
        # output.
        results = re.search(r'autograd profiler output.*(\n.*){6,100}', output)
        self.assertIsNotNone(results, self._fail_msg(
            'Distance between autograd prof output and end of output not in [6, 100] lines', output))

    def _check_cuda(self, output):
        if HAS_CUDA:
            results = re.search('CUDA mode', output)
            self.assertIsNotNone(results, self._fail_msg('Should tell users CUDA', output))
        else:
            results = re.search('CUDA mode', output)
            self.assertIsNone(results, self._fail_msg('Should not tell users about CUDA', output))

    @unittest.skipIf(HAS_CUDA, 'CPU-only test')
    def test_bottleneck_cpu_only(self):
        rc, out, err = self._run_bottleneck('bottleneck_test/test.py')
        self.assertEqual(rc, 0, 'Run failed with\n{}'.format(err))

        self._check_run_args()
        self._check_environment_summary(out)
        self._check_autograd_summary(out)
        self._check_cprof_summary(out)
        self._check_cuda(out)

    @unittest.skipIf(not HAS_CUDA, 'No CUDA')
    @skipIfRocm
    def test_bottleneck_cuda(self):
        rc, out, err = self._run_bottleneck('bottleneck_test/test_cuda.py')
        self.assertEqual(rc, 0, 'Run failed with\n{}'.format(err))

        self._check_run_args()
        self._check_environment_summary(out)
        self._check_autograd_summary(out)
        self._check_cprof_summary(out)
        self._check_cuda(out)


from torch.utils.collect_env import get_pretty_env_info


class TestCollectEnv(TestCase):
    def test_smoke(self):
        info_output = get_pretty_env_info()
        self.assertTrue(info_output.count('\n') >= 17)


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


def sum_of_state_dict(state_dict):
    s = 0
    for _, v in state_dict.items():
        s += v.sum()
    return s

SUM_OF_HUB_EXAMPLE = 431080
TORCHHUB_EXAMPLE_RELEASE_URL = 'https://github.com/ailzhang/torchhub_example/releases/download/0.1/mnist_init_ones'

@unittest.skipIf(IS_SANDCASTLE, 'Sandcastle cannot ping external')
class TestHub(TestCase):
    @retry(HTTPError, tries=3)
    def test_load_from_github(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    @retry(HTTPError, tries=3)
    def test_load_from_branch(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example:ci/test_slash',
            'mnist',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    @retry(HTTPError, tries=3)
    def test_set_dir(self):
        temp_dir = tempfile.gettempdir()
        hub.set_dir(temp_dir)
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)
        assert os.path.exists(temp_dir + '/ailzhang_torchhub_example_master')
        shutil.rmtree(temp_dir + '/ailzhang_torchhub_example_master')

    @retry(HTTPError, tries=3)
    def test_list_entrypoints(self):
        entry_lists = hub.list('ailzhang/torchhub_example', force_reload=True)
        self.assertObjectIn('mnist', entry_lists)

    @retry(HTTPError, tries=3)
    def test_download_url_to_file(self):
        temp_file = os.path.join(tempfile.gettempdir(), 'temp')
        hub.download_url_to_file(TORCHHUB_EXAMPLE_RELEASE_URL, temp_file, progress=False)
        loaded_state = torch.load(temp_file)
        self.assertEqual(sum_of_state_dict(loaded_state),
                         SUM_OF_HUB_EXAMPLE)

    @retry(HTTPError, tries=3)
    def test_load_state_dict_from_url(self):
        loaded_state = hub.load_state_dict_from_url(TORCHHUB_EXAMPLE_RELEASE_URL)
        self.assertEqual(sum_of_state_dict(loaded_state),
                         SUM_OF_HUB_EXAMPLE)

    @retry(HTTPError, tries=3)
    def test_load_zip_checkpoint(self):
        hub_model = hub.load(
            'ailzhang/torchhub_example',
            'mnist_zip',
            pretrained=True,
            verbose=False)
        self.assertEqual(sum_of_state_dict(hub_model.state_dict()),
                         SUM_OF_HUB_EXAMPLE)

    @unittest.skipIf(PY2, "Requires python 3")
    def test_hub_dir(self):
        with tempfile.TemporaryDirectory('hub_dir') as dirname:
            torch.hub.set_dir(dirname)
            self.assertEqual(torch.hub._get_torch_home(), dirname)


class TestHipify(TestCase):
    def test_import_hipify(self):
        from torch.utils.hipify import hipify_python # noqa


if __name__ == '__main__':
    run_tests()
