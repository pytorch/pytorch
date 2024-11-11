# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import TestCase
from torch.ao.quantization.utils import get_fqn_to_example_inputs
from torch.ao.nn.quantized.modules.utils import _quantize_weight
from torch.ao.quantization import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver


class TestUtils(TestCase):
    def _test_get_fqn_to_example_inputs(self, M, example_inputs, expected_fqn_to_dim):
        m = M().eval()
        fqn_to_example_inputs = get_fqn_to_example_inputs(m, example_inputs)
        for fqn, expected_dims in expected_fqn_to_dim.items():
            assert fqn in expected_fqn_to_dim
            example_inputs = fqn_to_example_inputs[fqn]
            for example_input, expected_dim in zip(example_inputs, expected_dims):
                assert example_input.dim() == expected_dim

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
        assert "sub" in fqn_to_example_inputs
        assert isinstance(fqn_to_example_inputs["sub"][1], list)
        assert isinstance(fqn_to_example_inputs["sub"][2], dict) and \
            "3" in fqn_to_example_inputs["sub"][2]

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
        assert observer.min_val == fp_min
        assert observer.max_val == fp_max

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

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
        assert observer.min_val == fp_min
        assert observer.max_val == fp_max

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

        # Actual weight values can be outside than observer [min_val, max_val] for the moving average observer
        float_tensor *= 1.2

        quantized_tensor = _quantize_weight(float_tensor, observer)
        assert quantized_tensor.int_repr().max().item() == q8_max
        assert quantized_tensor.int_repr().min().item() == q8_min

    def test_uint4_int4_dtype(self):

        def up_size(size):
            return (*size[:-1], size[-1] * 2)

        for dtype in [torch.uint4, torch.int4]:
            class UInt4OrInt4Tensor(torch.Tensor):
                @staticmethod
                def __new__(cls, elem, **kwargs):
                    assert elem.dtype is torch.uint8
                    assert not kwargs.get("requires_grad", False)
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
            assert x.dtype == dtype
