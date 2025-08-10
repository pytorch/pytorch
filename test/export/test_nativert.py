# Owner(s): ["oncall: export"]


import copy
import pathlib
import tempfile
import unittest

import torch
from torch._C._nativert import PyModelRunner
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils import _pytree as pytree


try:
    from . import test_export, testing
except ImportError:
    import test_export
    import testing

from torch.export import export


test_classes = {}


def _use_real_inputs(ep):
    ep = copy.copy(ep)

    has_fake_tensor = False

    def _to_real_tensor(t):
        if isinstance(t, torch.nn.Parameter):
            return torch.nn.Parameter(_to_real_tensor(t.data))
        if isinstance(t, FakeTensor):
            nonlocal has_fake_tensor
            has_fake_tensor = True
            return torch.randn(t.shape, device=t.device, requires_grad=t.requires_grad)
        return t

    new_example_inputs = pytree.tree_map_only(
        (torch.Tensor, torch.nn.Parameter), _to_real_tensor, ep.example_inputs
    )
    if has_fake_tensor:
        ep.example_inputs = new_example_inputs

    ep = ep._update(
        ep.graph_module,
        ep.graph_signature,
        state_dict=pytree.tree_map_only(
            (torch.Tensor, torch.nn.Parameter), _to_real_tensor, ep.state_dict
        ),
        constants=pytree.tree_map_only(
            (torch.Tensor, torch.nn.Parameter), _to_real_tensor, ep.constants
        ),
    )
    return ep


def _is_supported_types(arg) -> bool:
    if isinstance(arg, list):
        return (
            all(_is_supported_types(a) for a in arg)
            and len({type(a) for a in arg}) <= 1
        )
    elif isinstance(arg, tuple):
        return all(_is_supported_types(a) for a in arg)
    elif isinstance(arg, dict):
        return (
            all(_is_supported_types(a) for a in arg.values())
            and len({type(a) for a in arg.values()}) <= 1
        )
    elif isinstance(arg, (torch.Tensor, int, float, bool, str)):
        return True
    elif arg is None:
        return True
    else:
        return False


def run_with_nativert(ep):
    # Downstream tests might mutate the exported program in subtle ways, so
    # we need to make a copy here.
    ep_infer = copy.deepcopy(ep)
    ep_infer = _use_real_inputs(ep_infer.run_decompositions())
    MODEL_NAME = "forward"

    # TODO Does named tempfile have collision?
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.export.pt2_archive._package.package_pt2(
            f, exported_programs={MODEL_NAME: ep_infer}
        )
        filename = f.name

    try:
        ep_args, ep_kwargs = ep_infer.example_inputs
        ep_args_copied, ep_kwargs_copied = (
            copy.deepcopy(ep_args),
            copy.deepcopy(ep_kwargs),
        )
        torch.manual_seed(0)
        try:
            flat_expected = pytree.tree_leaves(
                ep_infer.module()(*ep_args_copied, **ep_kwargs_copied)
            )
        except Exception as e:
            raise unittest.case.SkipTest(str(e)) from e

        model_runner = PyModelRunner(filename, MODEL_NAME)
        torch.manual_seed(0)
        if _is_supported_types((ep_args, ep_kwargs)):
            results = model_runner.run(*ep_args, **ep_kwargs)
        else:
            results = model_runner.run_with_flat_inputs_and_outputs(
                *pytree.tree_leaves((ep_args, ep_kwargs))
            )
        flat_results = pytree.tree_leaves(results)
        assert len(flat_results) == len(flat_expected)
        for result, expected in zip(flat_results, flat_expected):
            assert type(result) == type(expected)
            if isinstance(result, torch.Tensor) and isinstance(expected, torch.Tensor):
                assert result.shape == expected.shape
                assert result.dtype == expected.dtype
                assert result.device == expected.device
                torch.testing.assert_close(result, expected, equal_nan=True)
            else:
                assert result == expected
    except RuntimeError as e:
        # User need to register pytree type on the cpp side, which
        # cannot be tested in python unittest.
        if "Unknown pytree node type" in str(e):
            pass
        else:
            raise e
    finally:
        pathlib.Path(filename).unlink(missing_ok=True)
    return ep


def mocked_nativert_export_strict(*args, **kwargs):
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=True)

    run_with_nativert(ep)
    return ep


def mocked_nativert_export_nonstrict(*args, **kwargs):
    if "strict" in kwargs:
        ep = export(*args, **kwargs)
    else:
        ep = export(*args, **kwargs, strict=False)

    run_with_nativert(ep)
    return ep


def make_dynamic_cls(cls, strict=False):
    cls_prefix = "NativeRT"

    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            cls_prefix,
            test_export.CPP_RUNTIME_STRICT_SUFFIX,
            mocked_nativert_export_strict,
            xfail_prop="_expected_failure_cpp_runtime",
            test_only_if_no_xfail=True,
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            cls_prefix,
            test_export.CPP_RUNTIME_NONSTRICT_SUFFIX,
            mocked_nativert_export_nonstrict,
            xfail_prop="_expected_failure_cpp_runtime_non_strict",
            test_only_if_no_xfail=True,
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__


tests = [
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test, strict=True)
    make_dynamic_cls(test, strict=False)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
