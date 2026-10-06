# Owner(s): ["module: linear algebra"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import (
    _f32_to_e8m0_rceil,
    compute_error,
    from_blocked_format,
    to_mxfp,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.mxfp8_test_utils import (
    assert_mxfp8_semantics,
    make_f32_to_e8m0_rceil_cases,
    make_mxfp8_semantic_cases,
)


class TestMXFP8ReferenceNumerics(TestCase):
    def test_f32_to_e8m0_rceil(self, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L113
        values, expected = make_f32_to_e8m0_rceil_cases(device=device)
        self.assertEqual(_f32_to_e8m0_rceil(values), expected.to(device))

    @parametrize("input_dtype", (torch.float32, torch.bfloat16))
    def test_mxfp8_corner_case_bytes(self, input_dtype, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L264
        cases = make_mxfp8_semantic_cases(input_dtype, device=device)
        scales, qdata = to_mxfp(cases.inputs, format="mxfp8")
        assert_mxfp8_semantics(qdata, scales, cases)

    @parametrize("input_dtype", (torch.float32, torch.bfloat16))
    def test_to_mx_rceil_randn_sqnr(self, input_dtype, device):
        data_hp = torch.randn(128, 128, device=device, dtype=input_dtype)
        scales, qdata = to_mxfp(data_hp, format="mxfp8")
        original = data_hp.float()
        dequantized = from_blocked_format(qdata, scales).float()
        sqnr = compute_error(original, dequantized)
        self.assertGreater(sqnr.item(), 18.0)


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
