# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase
from torch.ao.quantization.utils import get_fqn_to_example_inputs
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class TestUtils(TestCase):
    def _test_get_fqn_to_example_inputs(self, M, example_inputs, expected_fqn_to_dim):
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        for fqn, expected_dims in expected_fqn_to_dim.items():
            if fqn not in expected_fqn_to_dim:
                raise AssertionError(f"Expected fqn {fqn} in expected_fqn_to_dim")
            example_inputs = fqn_to_example_inputs[fqn]
            for example_input, expected_dim in zip(example_inputs, expected_dims):
                if example_input.dim() != expected_dim:
                    raise AssertionError(
                        f"Expected dim {expected_dim}, got {example_input.dim()}"
                    )

    def test_get_fqn_to_example_inputs_simple(self):
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        expected_fqn_to_dim = {
            "": (2,),
            "linear1": (2,),
            "linear2": (2,),
            "sub": (2,),
            "sub.linear1": (2,),
            "sub.linear2": (2,)
        }
        example_inputs = (torch.rand(1, 5),)
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_default_kwargs(self):
        """ Test that we can get example inputs for functions with default keyword arguments
        """
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x, key1=torch.rand(1), key2=torch.rand(1)):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                # only override `key2`, `key1` will use default
                x = self.sub(x, key2=torch.rand(1, 2))
                return x

        expected_fqn_to_dim = {
            "": (2,),
            "linear1": (2,),
            "linear2": (2,),
            # second arg is `key1`, which is using default argument
            # third arg is `key2`, override by callsite
            "sub": (2, 1, 2),
            "sub.linear1": (2,),
            "sub.linear2": (2,)
        }
        example_inputs = (torch.rand(1, 5),)
        self._test_get_fqn_to_example_inputs(M, example_inputs, expected_fqn_to_dim)

    def test_get_fqn_to_example_inputs_complex_args(self):
        """ Test that we can record complex example inputs such as lists and dicts
        """
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)

            def forward(self, x, list_arg, dict_arg):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.sub = Sub()

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x, [x], {"3": x})
                return x

        example_inputs = (torch.rand(1, 5),)
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        if "sub" not in fqn_to_example_inputs:
            raise AssertionError("Expected 'sub' in fqn_to_example_inputs")
        if not isinstance(fqn_to_example_inputs["sub"][1], list):
            raise AssertionError("Expected fqn_to_example_inputs['sub'][1] to be list")
        if not isinstance(fqn_to_example_inputs["sub"][2], dict):
            raise AssertionError("Expected fqn_to_example_inputs['sub'][2] to be dict")
        if "3" not in fqn_to_example_inputs["sub"][2]:
            raise AssertionError("Expected '3' in fqn_to_example_inputs['sub'][2]")

    def test_quantize_weight_clamping_per_tensor(self):
        """ Test quant_{min, max} from per tensor observer is honored by `_quantize_weight` method
        """
        fp_min, fp_max = -1000.0, 1000.0
        q8_min, q8_max = -10, 10

        float_tensor = torch.tensor([fp_min, fp_max])

        observer = MovingAverageMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_tensor_symmetric,
        )

        observer(float_tensor)
        if observer.min_val != fp_min:
            raise AssertionError(f"Expected min_val {fp_min}, got {observer.min_val}")
        if observer.max_val != fp_max:
            raise AssertionError(f"Expected max_val {fp_max}, got {observer.max_val}")

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

    def test_quantize_weight_clamping_per_channel(self):
        """ Test quant_{min, max} from per channel observer is honored by `_quantize_weight` method
        """
        fp_min, fp_max = -1000.0, 1000.0
        q8_min, q8_max = -10, 10

        float_tensor = torch.tensor([[fp_min, fp_max]])

        observer = MovingAveragePerChannelMinMaxObserver(
            averaging_constant=1.0,
            dtype=torch.qint8,
            quant_min=q8_min,
            quant_max=q8_max,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )

        observer(float_tensor)
        if observer.min_val != fp_min:
            raise AssertionError(f"Expected min_val {fp_min}, got {observer.min_val}")
        if observer.max_val != fp_max:
            raise AssertionError(f"Expected max_val {fp_max}, got {observer.max_val}")

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        if quantized_tensor.int_repr().max().item() != q8_max:
            raise AssertionError(
                f"Expected max {q8_max}, got {quantized_tensor.int_repr().max().item()}"
            )
        if quantized_tensor.int_repr().min().item() != q8_min:
            raise AssertionError(
                f"Expected min {q8_min}, got {quantized_tensor.int_repr().min().item()}"
            )

    def test_uint4_int4_dtype(self):

        def up_size(size):
            return (*size[:-1], size[-1] * 2)

        for dtype in [torch.uint4, torch.int4]:
            class UInt4OrInt4Tensor(torch.Tensor):
                @staticmethod
                def __new__(cls, elem, **kwargs):
                    if elem.dtype is not torch.uint8:
                        raise AssertionError(f"Expected dtype uint8, got {elem.dtype}")
                    if kwargs.get("requires_grad", False):
                        raise AssertionError("Expected requires_grad to be False")
                    kwargs["requires_grad"] = False
                    return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=dtype, **kwargs)

                def __init__(self, elem):
                    self.elem = elem

                @classmethod
                def __torch_dispatch__(cls, func, types, args, kwargs=None):
                    pass

            # make sure it runs
            x = UInt4OrInt4Tensor(torch.tensor([
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
                [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF],
            ], dtype=torch.uint8))
            if x.dtype != dtype:
                raise AssertionError(f"Expected dtype {dtype}, got {x.dtype}")

if __name__ == "__main__":
    raise_on_run_directly("test/test_quantization.py")
