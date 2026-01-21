# Owner(s): ["oncall: export"]


import copy
import tempfile
import unittest

from parameterized import parameterized

import torch
import torch._dynamo as torchdynamo
from torch._C._nativert import PyModelRunner
from torch._dynamo.test_case import TestCase
from torch._environment import is_fbcode
from torch._subclasses.fake_tensor import FakeTensor
from torch.nativert.backends._lower_utils import (
    lower_exported_program,
    package_nativert_with_aoti_delegate,
)
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
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
    with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
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
                assert type(result) is type(expected)
                if isinstance(result, torch.Tensor) and isinstance(
                    expected, torch.Tensor
                ):
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


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
@unittest.skipIf(not is_fbcode(), "FBcode only for now")
class TestNativeRT(TestCase):
    @staticmethod
    def get_module():
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        return M()

    @staticmethod
    def get_module_multi_output():
        class MMultiOutput(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return (self.relu(self.linear(x)), x)

        return MMultiOutput()

    @staticmethod
    def get_model_pytree():
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)

            def forward(self, x):
                x1, (x2, x3) = x
                y1 = self.linear1(x1)
                y2 = self.linear2(x2)
                y3 = self.linear2(x3)
                return (y1, (y2, y3))

        return M()

    parameters = []
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not HAS_CUDA_AND_TRITON:
            continue
        for module, sample_inputs in [
            (get_module.__func__().to(device), (torch.randn(4, 4).to(device),)),
            (
                get_module_multi_output.__func__().to(device),
                (torch.randn(4, 4).to(device),),
            ),
            (
                get_model_pytree.__func__().to(device),
                (
                    (
                        torch.randn(4, 4).to(device),
                        (
                            torch.randn(4, 4).to(device),
                            torch.randn(4, 4).to(device),
                        ),
                    ),
                ),
            ),
        ]:
            parameters.append(
                (
                    device,
                    module,
                    sample_inputs,
                )
            )

    @parameterized.expand(parameters)
    def test_aoti(self, device, m, sample_inputs):
        MODEL_NAME = "model"
        BACKEND_ID = "aoti"

        # get the original EP
        original_ep = torch.export.export(m, sample_inputs)

        aoti_delegate_ep, aoti_files = lower_exported_program(
            original_ep, MODEL_NAME, BACKEND_ID
        )

        # package everything needed for the NativeRT to execute the AOTI delegate
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            package_nativert_with_aoti_delegate(
                f,
                MODEL_NAME,
                BACKEND_ID,
                original_ep,
                aoti_delegate_ep,
                aoti_files,
            )
            filename = f.name

            try:
                ep_args, ep_kwargs = aoti_delegate_ep.example_inputs
                ep_args_copied, ep_kwargs_copied = (
                    copy.deepcopy(ep_args),
                    copy.deepcopy(ep_kwargs),
                )
                torch.manual_seed(0)
                try:
                    flat_expected = pytree.tree_leaves(
                        aoti_delegate_ep.module()(*ep_args_copied, **ep_kwargs_copied)
                    )
                except Exception as e:
                    raise unittest.case.SkipTest(str(e)) from e

                model_runner = PyModelRunner(filename, f"{MODEL_NAME}-{BACKEND_ID}")
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
                    assert type(result) is type(expected)
                    if isinstance(result, torch.Tensor) and isinstance(
                        expected, torch.Tensor
                    ):
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


if is_fbcode():
    for test in [test_export.TestExport]:
        make_dynamic_cls(test, strict=True)
        make_dynamic_cls(test, strict=False)
    del test


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # nativert has not been supported on XPU yet.
    if not torch.xpu.is_available():
        run_tests()
