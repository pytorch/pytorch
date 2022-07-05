# Owner(s): ["oncall: fx"]

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import unittest
import warnings
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_methods_invocations import DecorateInfo
from torch.testing._internal.common_methods_invocations import op_db, wrapper_set_seed

from torch.testing._internal.common_device_type import ops
from torch.fx.experimental.proxy_tensor import make_fx

# Copied from functorch
def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, True)


def skip(op_name, variant_name='', *, device_type=None, dtypes=None):
    return (op_name, variant_name, device_type, dtypes, False)


def skipOps(test_case_name, base_test_name, to_skip):
    all_opinfos = op_db
    for xfail in to_skip:
        op_name, variant_name, device_type, dtypes, expected_failure = xfail
        matching_opinfos = [o for o in all_opinfos
                            if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure,
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip("Skipped!"),
                                         test_case_name, base_test_name,
                                         device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        return fn
    return wrapped


USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)


class TestProxyTensor(TestCase):
    def test_make_fx(self, device):
        def f(x):
            return torch.sin(x)
        inp = torch.randn(3)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_scalar_device(self, device):
        def f(a, b):
            return a + b
        inps = [torch.randn(3, device=device), torch.tensor(5)]
        fx_f = make_fx(f)(*inps)
        self.assertEqual(fx_f(*inps), f(*inps))


    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_resnet18_backward_trace(self, device):
        mod = torchvision.models.resnet18()

        def f(x):
            out = mod(x)
            out.sum().backward()
            return [a.grad for a in mod.parameters()]

        inp = torch.randn(3, 3, 250, 250, requires_grad=True)
        grads = f(inp)

        mod.zero_grad()
        mod(inp).sum().backward()
        grads2 = [a.grad for a in mod.parameters()]
        self.assertEqual(grads, grads2)

    def test_proxy_tensor(self):
        def f_grad(x):
            val = x.cos().cos().sum()
            return torch.autograd.grad(val, x)

        def f_backward(x):
            val = x.cos().cos().sum()
            val.backward()
            return x.grad

        for f in [f_grad, f_backward]:
            traced_graph = make_fx(f)(torch.randn(3, requires_grad=True))
            inp = torch.randn(3, requires_grad=True)
            traced_graph_out = traced_graph(inp)
            assert inp.grad is None
            torch.testing.assert_close(traced_graph_out, f(inp))

    def test_mode_tracing_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        # default behavior should trace factory functions
        traced = make_fx(f)(torch.randn(3))
        self.assertTrue(
            any(
                isinstance(node.target, torch._ops.OpOverloadPacket) and node.target._qualified_op_name == 'aten::randn'
                for node in traced.graph.nodes
            )
        )

    def test_mode_tracing_factory_function_no_factory_function(self):
        def f(x):
            return x + torch.randn(x.shape)

        traced = make_fx(f, trace_factory_functions=False)(torch.randn(3))  # default behavior should not trace factory functions
        self.assertFalse(
            any(
                isinstance(node.target, torch._ops.OpOverloadPacket) and node.target._qualified_op_name == 'aten::randn'
                for node in traced.graph.nodes
            )
        )

make_fx_failures = {
    xfail('allclose'),
    xfail('equal'),
    xfail('linalg.eigvals'),
    xfail('nn.functional.max_pool1d', device_type='cpu'),
    # empty
    skip('new_empty'),
    skip('empty_like'),
    skip('empty'),
    # flaky
    skip('linalg.lstsq', 'grad_oriented'),
    skip('nn.functional.max_unpool1d', '', device_type='cpu'),
    skip('nn.functional.max_unpool2d', '', device_type='cpu'),
    skip('nn.functional.max_unpool3d', '', device_type='cpu'),
    skip('linalg.lstsq'),  # flaky, probably just a precision issue
    xfail('histogram'),
    xfail('scatter'),
    # data-dependent control flow
    xfail('cov'),
    xfail('istft'),
    xfail('nanquantile'),
    xfail('nn.functional.gaussian_nll_loss'),
    xfail('quantile'),
    xfail('tensor_split'),
    xfail('corrcoef'),
    # Masked failures (creating a scalar tensor just to call `.item` on it)
    xfail('_masked.amax'),
    xfail('_masked.amax'),
    xfail('_masked.amin'),
    xfail('_masked.argmax'),
    xfail('_masked.argmin'),
    xfail('_masked.cumprod'),
    xfail('_masked.cumsum'),
    xfail('_masked.log_softmax'),
    xfail('_masked.logaddexp'),
    xfail('_masked.logsumexp'),
    xfail('_masked.mean'),
    xfail('_masked.median'),
    xfail('_masked.norm'),
    xfail('_masked.prod'),
    xfail('_masked.softmax'),
    xfail('_masked.softmin'),
    xfail('_masked.std'),
    xfail('_masked.sum'),
    xfail('_masked.var'),

    # Seems like it's creating a sparse tensor that isn't captured by tensor.is_sparse
    xfail('sparse.sampled_addmm'),

    # Seems like it's creating a sparse tensor that isn't captured by tensor.is_sparse
    xfail('nn.functional.ctc_loss'),
}


class TestProxyTensorOpInfo(TestCase):
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestProxyTensorOpInfo', 'test_make_fx_exhaustive', make_fx_failures
             )
    def test_make_fx_exhaustive(self, device, dtype, op):

        def f(args, kwargs):
            return op.op(*args, **kwargs)
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        new_f = None
        for sample_input in sample_inputs_itr:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            new_f = make_fx(f, trace_factory_functions=True)(args, kwargs)
            for arg in args:
                if isinstance(arg, torch.Tensor) and arg.dtype == torch.float:
                    arg.uniform_(0, 1)
            try:
                old_out = f(args, kwargs)
            except Exception:
                continue
            new_out = wrapper_set_seed(new_f, args, kwargs)
            self.assertEqual(new_out, old_out)



only_for = ("cpu")
instantiate_device_type_tests(
    TestProxyTensor,
    globals(),
    only_for=only_for,
)
instantiate_device_type_tests(TestProxyTensorOpInfo, globals(), only_for=only_for)


if __name__ == '__main__':
    run_tests()
