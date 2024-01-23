# Owner(s): ["module: nn"]
from copy import deepcopy
from itertools import product
from tempfile import NamedTemporaryFile
import unittest

import torch
import torch.nn as nn
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import TestCase, \
    TEST_NUMPY, IS_WINDOWS, skipIfTorchDynamo, instantiate_parametrized_tests, \
    run_tests, wrapSwapTensorsTest, skipIfCrossRef
from torch.utils._pytree import tree_map

if TEST_NUMPY:
    import numpy as np


class TestLoadStateDict(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @unittest.skipIf(not TEST_NUMPY, "numpy not found")
    def test_load_state_dict_invalid(self):
        m = torch.nn.Linear(2, 2, bias=False)

        state_dict = {'weight': np.random.randn(2, 2)}
        with self.assertRaisesRegex(RuntimeError,
                                    "expected torch.Tensor or Tensor-like object from checkpoint but received"):
            m.load_state_dict(state_dict)

        state_dict = {'weight': ((1., 1.), (2., 2.))}
        with self.assertRaisesRegex(RuntimeError,
                                    "expected torch.Tensor or Tensor-like object from checkpoint but received"):
            m.load_state_dict(state_dict)

    def test_load_state_dict_type(self):
        m = nn.Module()

        with self.assertRaisesRegex(TypeError,
                                    "Expected state_dict to be dict-like, got"):
            m.load_state_dict("")
        with self.assertRaisesRegex(TypeError,
                                    "Expected state_dict to be dict-like, got"):
            m.load_state_dict(2)

    def test_load_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv1 = nn.Conv2d(3, 3, 3, bias=True)
        block.conv2 = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)
        conv1_bias_dtype = block.conv1.bias.dtype

        state_dict = net.state_dict()
        state_dict.update({
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'bn.running_mean': torch.randn(2),
        })
        # Also test if a DDP state_dict can be loaded from a local model.
        ddp_state_dict = net.state_dict()
        ddp_state_dict.update({
            'module.linear1.weight': torch.ones(5, 5),
            'module.block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'module.bn.running_mean': torch.randn(2),
        })
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ddp_state_dict, 'module.')
        for sd in [state_dict, ddp_state_dict]:
            incompatible_keys = net.load_state_dict(sd)
            self.assertEqual(len(incompatible_keys.missing_keys), 0)
            self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
            self.assertNotIn('Incompatible', str(incompatible_keys))
            self.assertEqual(net.linear1.weight, sd['linear1.weight'])
            self.assertEqual(net.block.conv1.bias, sd['block.conv1.bias'])
            self.assertEqual(net.bn.running_mean, sd['bn.running_mean'])

        state_dict = net.state_dict()
        state_dict.update({'extra': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra', incompatible_keys.unexpected_keys)
        self.assertIn('Incompatible', str(incompatible_keys))

        state_dict = net.state_dict()
        state_dict.update({'extra.param': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 0)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('extra.param', incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        del state_dict['linear1.weight']
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 0)
        self.assertIn('linear1.weight', incompatible_keys.missing_keys)
        state_dict.update({'extra.param': torch.ones(5)})
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        incompatible_keys = net.load_state_dict(state_dict, strict=False)
        self.assertEqual(len(incompatible_keys.missing_keys), 1)
        self.assertEqual(len(incompatible_keys.unexpected_keys), 1)
        self.assertIn('linear1.weight', incompatible_keys.missing_keys)
        self.assertIn('extra.param', incompatible_keys.unexpected_keys)

        state_dict = net.state_dict()
        state_dict.update({'bn.running_mean': torch.rand(14, 4)})  # wrong size
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict))
        self.assertRaises(RuntimeError, lambda: net.load_state_dict(state_dict, strict=False))

        state_dict = net.state_dict()
        old_state_dict = deepcopy(state_dict)
        state_dict = {
            'linear1.weight': torch.ones(5, 5),
            'block.conv1.bias': torch.arange(1, 4, dtype=conv1_bias_dtype),
            'bn.running_mean': torch.randn(2),
            'nonexistent_key': torch.rand(3)
        }
        net.load_state_dict(state_dict, strict=False)
        self.assertEqual(net.linear1.weight, state_dict['linear1.weight'])
        self.assertEqual(net.block.conv1.bias, state_dict['block.conv1.bias'])
        self.assertEqual(net.bn.running_mean, state_dict['bn.running_mean'])
        new_state_dict = net.state_dict()
        del old_state_dict['linear1.weight']
        del old_state_dict['block.conv1.bias']
        del old_state_dict['bn.running_mean']
        for k, v, in old_state_dict.items():
            self.assertTrue(v.equal(new_state_dict[k]))

    def test_load_state_dict_BC(self):
        # BatchNormNd
        # Added num_batches_tracked buffer at version 2. For state dict with
        # earlier versions or no versions, it should provide default value of 0.
        bn = nn.BatchNorm2d(3)
        state_dict = bn.state_dict()
        del state_dict['num_batches_tracked']
        state_dict._metadata['']['version'] = 1  # version 1
        bn.load_state_dict(state_dict)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        self.assertEqual(bn.num_batches_tracked.item(), 0)
        del state_dict._metadata['']['version']  # no version
        bn.load_state_dict(state_dict)
        self.assertEqual(bn.num_batches_tracked.dtype, torch.long)
        self.assertEqual(bn.num_batches_tracked.item(), 0)

    def test_load_state_dict_child(self):
        base_module = nn.Linear(1, 1)
        model = base_module
        for _ in range(3):
            model = nn.Sequential(*[deepcopy(model) for _ in range(10)])

        def hook_fn(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
            module_state_dict = module.state_dict()
            self.assertEqual(len(module_state_dict.keys()), len(state_dict.keys()))

        model[0][0]._register_load_state_dict_pre_hook(hook_fn, with_module=True)
        model.load_state_dict(model.state_dict(), strict=True)

    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    def test_register_state_dict_pre_hook_backward_compat(self):
        called = False

        def my_state_dict_pre_hook(*args, **kwargs):
            nonlocal called
            called = True

        m = nn.Linear(1, 1)
        self.assertTrue(hasattr(m, '_state_dict_pre_hooks'))
        delattr(m, '_state_dict_pre_hooks')
        # Save and load, ensure we can still call state_dict
        # without running into issues.
        with NamedTemporaryFile() as f:
            # Note that torch.save / torch.load is not recommended
            # to save / load modules.
            torch.save(m, f.name)
            m = torch.load(f.name)

        # Ensure we can run state_dict without issues
        _ = m.state_dict()
        self.assertFalse(called)
        m.register_state_dict_pre_hook(my_state_dict_pre_hook)
        _ = m.state_dict()
        self.assertTrue(called)

    # fails in crossref mode when swapping tensors as LSTM installs weakrefs
    @skipIfCrossRef
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    def test_load_state_dict_ref_cycle(self):
        # load_state_dict shouldn't cause a reference cycle involving Tensors
        import gc

        m = torch.nn.LSTM(16, 16, bidirectional=True)

        gc.collect()
        m.load_state_dict(deepcopy(m).state_dict())
        refcycles = gc.collect()

        self.assertEqual(refcycles, 0)

    def test_load_state_dict_custom(self):

        class CustomState(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(1))
                self.sub = torch.nn.Linear(5, 5)

            def _save_to_state_dict(self, destination, prefix, keep_vars):
                destination[prefix + "serialized"] = self.param.data + 1

            def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs):
                # skip some of the error handling
                self.param.data.copy_(state_dict[prefix + "serialized"] - 1)

        # use sequential to verify nesting
        m = nn.Sequential(CustomState())
        with torch.no_grad():
            m[0].param[0] = 10
            m[0].sub.weight[0, 0] = 555
        state_dict = m.state_dict()
        self.assertEqual(state_dict["0.serialized"].item(), 11)
        self.assertIn("0.sub.weight", state_dict)
        self.assertNotIn("0.param", state_dict)
        del m
        mm = nn.Sequential(CustomState())
        self.assertEqual(mm[0].param[0].item(), 1)
        mm.load_state_dict(state_dict)
        self.assertEqual(mm[0].param[0].item(), 10)
        self.assertEqual(mm[0].sub.weight[0, 0].item(), 555)

    def test_load_state_dict_assign_meta(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, input):
                return self.bn(self.fc1(input))

        net = MyModule()
        state_dict = net.state_dict(keep_vars=True)

        with torch.device('meta'):
            net_meta = MyModule()

        net_meta.load_state_dict(state_dict, assign=True)

        # Make sure parameters and persistent buffers were assigned
        net_meta_state_dict = net_meta.state_dict(keep_vars=True)
        for key in state_dict.keys():
            if isinstance(state_dict[key], torch.nn.Parameter):
                self.assertTrue(state_dict[key] is net_meta_state_dict[key])

        # Make sure that ordering of parameters and buffers is preserved
        net_named_parameters = net.named_parameters()
        net_named_buffers = net.named_buffers()
        net_meta_named_parameters = net_meta.named_parameters()
        net_meta_named_buffers = net_meta.named_buffers()

        for p1, p2 in zip(net_named_parameters, net_meta_named_parameters):
            n1, _ = p1
            n2, _ = p2
            self.assertEqual(n1, n2)

        for p1, p2 in zip(net_named_buffers, net_meta_named_buffers):
            n1, _ = p1
            n2, _ = p2
            self.assertEqual(n1, n2)

        # Make sure outputs are the same
        t = torch.randn(4, 3)
        out_net = net(t)
        out_net_meta = net_meta(t.clone())

        self.assertEqual(out_net, out_net_meta)

    def test_load_state_dict_assign_with_optimizer(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, input):
                return self.bn(self.fc1(input))

        net = MyModule()
        opt = torch.optim.Adam(net.parameters(), lr=1000)
        x = torch.randn(4, 3)
        num_iters = 3

        for i in range(num_iters):
            opt.zero_grad()
            out = net(x)
            out.sum().backward()
            opt.step()

        opt_state_dict = deepcopy(opt.state_dict())
        net_state_dict = deepcopy(net.state_dict())

        with torch.device('meta'):
            net_meta = MyModule()

        net_meta.load_state_dict(net_state_dict, assign=True)
        # must create optimizer only after loading state_dict when assign=True
        opt2 = torch.optim.Adam(net_meta.parameters(), lr=1000)
        opt2.load_state_dict(opt_state_dict)

        y = x.clone()
        for i in range(num_iters):
            opt.zero_grad()
            out = net(x)
            out.sum().backward()
            opt.step()

            opt2.zero_grad()
            out2 = net_meta(y)
            out2.sum().backward()
            opt2.step()

        self.assertEqual(opt.state_dict(), opt2.state_dict())
        self.assertEqual(net.state_dict(), net_meta.state_dict())

    def test_load_state_dict_assign_shape_stride(self):
        # Assigned tensor is allowed to have different properties than initial
        # tensor except for shape
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.bn = nn.BatchNorm1d(5)

            def forward(self, input):
                return self.bn(self.fc1(input))

        net = MyModule()
        state_dict = net.state_dict()
        # loading should be ok if stride is different
        state_dict['fc1.weight'] = torch.randn(3, 5).transpose(0, 1)
        net2 = MyModule()
        net2.load_state_dict(state_dict, strict=False, assign=True)

        state_dict['fc1.weight'] = torch.randn(2, 4)
        with self.assertRaisesRegex(RuntimeError, "size mismatch for fc1.weight: copying a param with shape"):
            net2.load_state_dict(state_dict, strict=False, assign=True)

    def test_load_state_dict_warn_assign(self):
        with torch.device('meta'):
            m = torch.nn.Linear(3, 5)
        state_dict = m.state_dict()
        state_dict['weight'] = torch.empty_like(state_dict['weight'], device='cpu')
        with self.assertWarnsRegex(UserWarning, "for weight: copying from a non-meta parameter in the checkpoint to a meta"):
            m.load_state_dict(state_dict)


class MyLoadTensor(torch.Tensor):

    # enable deepcopy
    def new_empty(self, shape):
        return type(self)(shape)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        def module_load_from(dest, src):
            # always convert src to MyLoadTensor
            if type(src) is torch.Tensor:
                return MyLoadTensor(src)
            elif type(src) is MyLoadTensor:
                return src
            elif type(src) is MyWrapperLoadTensor:
                return MyLoadTensor(src._data)
            else:
                raise NotImplementedError(f"module_load_from(): unsupported type(src) {type(src)}")

        def module_load_to(src, dest):
            if type(dest) in {torch.nn.Parameter, torch.Tensor, MyLoadTensor, MyWrapperLoadTensor}:
                return src
            else:
                raise NotImplementedError(f"module_load_to(): unsupported type(dest) {type(dest)}")

        if func is torch.Tensor.module_load_from:
            return module_load_from(*args, **kwargs)
        elif func is torch.Tensor.module_load_to:
            return module_load_to(*args, **kwargs)
        elif func is torch.Tensor.detach:
            return MyLoadTensor(args[0].data.detach())
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


class MyWrapperLoadTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, data: torch.Tensor):
        t = torch.Tensor._make_wrapper_subclass(
            cls, data.size(),
            dtype=data.dtype, layout=data.layout,
            device=data.device, requires_grad=data.requires_grad,
            strides=data.stride(), storage_offset=data.storage_offset())
        return t

    def __init__(self, data: torch.Tensor):
        self._data = data

    def __repr__(self):
        return f"MyWrapperLoadTensor({self._data.__repr__()})"

    # enable deepcopy
    def new_empty(self, shape):
        return type(self)(self._data.new_empty(shape))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        def module_load_from(dest, src):
            # always convert src to MyWrapperLoadTensor
            if type(src) is torch.Tensor:
                return MyWrapperLoadTensor(src)
            elif type(src) is MyWrapperLoadTensor:
                return src
            elif type(src) is MyLoadTensor:
                return MyWrapperLoadTensor(src.data)
            else:
                raise NotImplementedError(f"module_load_from(): unsupported type(src) {type(src)}")

        def module_load_to(src, dest):
            if type(dest) in {torch.nn.Parameter, torch.Tensor, MyLoadTensor, MyWrapperLoadTensor}:
                return src
            else:
                raise NotImplementedError(f"module_load_to(): unsupported type(dest) {type(dest)}")

        if func is torch.Tensor.module_load_from:
            return module_load_from(*args, **kwargs)
        elif func is torch.Tensor.module_load_to:
            return module_load_to(*args, **kwargs)
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def unwrap(t):
            return t._data if isinstance(t, MyWrapperLoadTensor) else t

        kwargs = {} if kwargs is None else kwargs
        out = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        if isinstance(out, torch.Tensor):
            return MyWrapperLoadTensor(out)
        else:
            return out


class TestLoadStateDictSwap(TestCase):
    @skipIfCrossRef
    @skipIfTorchDynamo("Can't swap with dynamo as dynamo installs weakrefs")
    @wrapSwapTensorsTest(swap=True)
    def test_swap_subclass(self):

        def _create_model(subclass=None):
            m = torch.nn.Linear(2, 3, bias=False)
            m.register_buffer('buf', torch.randn(2, 3))
            if subclass is not None:
                m.weight = torch.nn.Parameter(subclass(m.weight))
                m.buf = subclass(m.buf)
            return m

        def _test(module_subclass=None, sd_subclass=None):
            m = _create_model(module_subclass)
            sd = _create_model(sd_subclass).state_dict()
            # deepcopy as swap_tensors might destructively modify state dict
            ref_sd = deepcopy(sd)
            m.load_state_dict(sd)
            self.assertEqual(m.weight, ref_sd['weight'])
            self.assertEqual(m.buf, ref_sd['buf'])
            self.assertTrue(isinstance(m.weight, torch.nn.Parameter))
            self.assertTrue(not isinstance(m.buf, torch.nn.Parameter))

            weight_type, buf_type = (torch.nn.Parameter, torch.Tensor)
            if sd_subclass is not None and module_subclass is not None:
                # param.module_load_from takes precedence over input.module_load_to
                weight_type, buf_type = (module_subclass, module_subclass)
            elif sd_subclass is not None:
                weight_type, buf_type = (sd_subclass, sd_subclass)
            elif module_subclass is not None:
                weight_type, buf_type = (module_subclass, module_subclass)
            self.assertTrue(type(m.weight) is weight_type)
            self.assertTrue(type(m.buf) is buf_type)

        subclasses = [None, MyLoadTensor, MyWrapperLoadTensor]
        for m_s, sd_s in product(subclasses, subclasses):
            _test(m_s, sd_s)


# run tests witsh swap enabled/disabled
def instantiate_parametrize_with_swap(cls):
    for attr_name in tuple(dir(cls)):
        class_attr = getattr(cls, attr_name)
        if attr_name.startswith('test_'):
            for swap in (True, False):
                setattr(cls, f'{attr_name}_swap_{swap}', wrapSwapTensorsTest(swap)(class_attr))
            delattr(cls, attr_name)


instantiate_parametrize_with_swap(TestLoadStateDict)
instantiate_parametrized_tests(TestLoadStateDict)
instantiate_parametrized_tests(TestLoadStateDictSwap)

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
