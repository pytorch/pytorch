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
        cases = make_mxfp8_semantic_cases(input_dtype, "rceil", device=device)
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

    def test_to_mx_rceil(self, device):
        # copied from
        # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/test/prototype/mx_formats/test_mx_tensor.py#L276
        # TODO(future PR): refactor below to make it look more like
        # `test_mxfp8_corner_case_bytes`

        # nan
        # fmt: off
        data_hp = torch.tensor(
            [
            2143289344, 1054459450, 1060527345, 1045656552, 1058239340, 1045057552, 1061158006, 1049626606,
            1052757568, 1032293288, 1056992320, 1064929425, 1061036255, 1047450552, 1057077424, 1055125012,
            1036491424, 1063542041, 1057099838, 1058731224, 1050189482, 1049114228, 1058347802, 1060065968,
            1058846156, 1048878912, 1065109089, 1054494928, 1044803976, 1049117692, 1065222528, 1056965012,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)

        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertTrue(torch.isnan(scales))
        # When any element in block is NaN, entire quantized block becomes NaN
        self.assertTrue(torch.all(torch.isnan(qdata)))
        # fp32 denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            6142315, 5096174, 3345704, 6178415, 5728750, 419002, 1716691, 4335089,
            5785800, 6234845, 1697524, 33075, 3975816, 3714822, 5411407, 3040844,
            7400945, 4474166, 7257182, 1273750, 5872176, 4694081, 2096530, 6273621,
            67028, 7585260, 4532315, 4599275, 6133942, 4542483, 5992199, 6862780,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # fmt: on
        ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # E8M0 byte 0 is 2^-127, so these FP32 subnormals remain representable
        # after scaling instead of being flushed to zero.
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            60, 58, 53, 60, 59, 29, 45, 56,
            59, 60, 45, 4, 55, 54, 58, 52,
            62, 57, 62, 42, 59, 57, 48, 60,
            8, 62, 57, 57, 60, 57, 59, 61,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))
        # bf16 denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            101, 3, 47, 54, 36, 19, 70, 79,
            35, 95, 28, 120, 84, 94, 20, 92,
            18, 42, 98, 58, 3, 26, 64, 86,
            60, 86, 52, 23, 61, 70, 59, 74,
            ],
            dtype=torch.uint16,
        ).view(torch.bfloat16)
        # fmt: on
        ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            61, 20, 52, 54, 49, 42, 57, 58,
            49, 60, 46, 63, 58, 60, 42, 60,
            41, 50, 60, 54, 20, 45, 56, 59,
            55, 59, 53, 44, 55, 57, 55, 57,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))
        # fp32 some denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            8388608, 1063716449, 1064039365, 1063568877, 1051091338, 1062185569, 1034449408, 1060813641,
            1054893736, 1034907680, 1036660744, 1023639888, 1058536559, 1050896496, 1049237634, 1064950601,
            1051852994, 1059794063, 1054011102, 1062023602, 1059467900, 1062276774, 1059155029, 1053287574,
            1064378711, 1055768540, 1045266076, 1059575077, 1054928758, 1040468200, 1058061961, 1053066436,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            0, 118, 119, 118, 106, 117, 91, 116,
            110, 91, 93, 80, 113, 106, 105, 120,
            107, 115, 109, 117, 114, 117, 114, 108,
            119, 111, 101, 114, 110, 96, 113, 108,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))
        # bf16 some denorm
        # fmt: off
        data_hp = torch.tensor(
            [
            128, 16118, 16143, 16074, 16187, 16002, 16193, 16217,
            15680, 16183, 16092, 16158, 16251, 15876, 15896, 16194,
            16135, 16214, 16205, 16110, 16122, 15960, 15824, 16106,
            16220, 16230, 15952, 15896, 16000, 16144, 16232, 16157,
            ],
            dtype=torch.uint16,
        ).view(torch.bfloat16)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            0, 111, 113, 109, 116, 104, 116, 118,
            84, 115, 110, 114, 120, 96, 98, 116,
            112, 117, 117, 111, 112, 102, 93, 111,
            118, 118, 101, 98, 104, 113, 118, 114,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))
        # zero
        data_hp = torch.tensor([0] * 32, dtype=torch.uint32).view(torch.float32)
        ground_truth_scale = torch.tensor([0], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        ground_truth_fp8 = torch.tensor([0] * 32, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))
        # fp32 normal
        # fmt: off
        data_hp = torch.tensor(
            [
            1037408064, 1058534842, 1053630662, 1063310394, 994704128, 1057245441, 1060663708, 1058053571,
            1052395648, 1064831570, 1038427336, 1064777688, 1059248393, 1060959028, 1062878286, 1057799482,
            1057854101, 1053562724, 1027482352, 1060498324, 1063238522, 1060472055, 1054346794, 1029092912,
            1056687298, 1059146141, 1037992128, 1064097772, 1056522806, 1059255744, 1064364912, 1060606252,
            ],
            dtype=torch.uint32,
        ).view(torch.float32)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            93, 113, 109, 118, 53, 112, 116, 113,
            108, 120, 94, 119, 114, 116, 118, 113,
            113, 109, 84, 115, 118, 115, 110, 85,
            112, 114, 94, 119, 112, 114, 119, 115,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))
        # bf16 normal
        # fmt: off
        data_hp = torch.tensor(
            [
            15752, 16143, 16182, 15896, 16195, 16186, 16048, 16223,
            15988, 16231, 16140, 16088, 16032, 16240, 16228, 16133,
            16210, 16024, 16248, 16187, 16050, 15696, 16060, 15956,
            16131, 16251, 15896, 16014, 15808, 16024, 16159, 16186,
            ],
            dtype=torch.uint16,
        ).view(torch.bfloat16)
        # fmt: on
        ground_truth_scale = torch.tensor([119], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        # fmt: off
        ground_truth_fp8 = torch.tensor(
            [
            88, 113, 115, 98, 116, 116, 107, 118,
            103, 118, 113, 110, 106, 119, 118, 112,
            117, 106, 120, 116, 107, 85, 108, 101,
            112, 120, 98, 105, 92, 106, 114, 116,
            ],
            dtype=torch.uint8,
        ).view(torch.float8_e4m3fn)
        # fmt: on
        scales, qdata = to_mxfp(data_hp.to(device), format="mxfp8")
        self.assertEqual(scales, ground_truth_scale.to(device))
        self.assertEqual(qdata, ground_truth_fp8.to(device))


instantiate_device_type_tests(TestMXFP8ReferenceNumerics, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
