# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Shared helpers for testing MXFP8 quantization semantics."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MXFP8SemanticCases:
    names: tuple[str, ...]
    inputs: torch.Tensor
    expected_data: torch.Tensor
    expected_scales: torch.Tensor


def make_mxfp8_semantic_cases(
    input_dtype: torch.dtype,
    *,
    device,
) -> MXFP8SemanticCases:
    r"""Create the shared bitwise contract for one-dimensional MXFP8 blocks."""
    # copied from
    # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/torchao/testing/_mxfp8_test_utils.py#L22
    if input_dtype not in (torch.float32, torch.bfloat16):
        raise AssertionError(f"unsupported input dtype: {input_dtype}")

    names = (
        "zero",
        "smallest_normal",
        "mixed_inf",
        "mixed_nan",
        "positive_fp8_max",
        "negative_fp8_max",
        "positive_dtype_max",
        "negative_dtype_max",
        "positive_smallest_subnormal",
        "negative_smallest_subnormal",
        "mixed_positive_underflow",
        "mixed_negative_underflow",
        "mixed_saturation_and_underflow",
        "rceil_premature_bf16_lowers_scale",
        "fp32_just_below_512",
        "all_nan",
        "positive_satfinite",
        "negative_satfinite",
    )
    if input_dtype == torch.float32:
        names += ("fp32_subnormals", "fp32_realistic_vals")
    else:
        names += ("bf16_subnormals", "bf16_realistic_vals")

    inputs = torch.empty((len(names), 32), dtype=input_dtype, device=device)
    expected_scales = torch.empty((len(names),), dtype=torch.uint8)
    expected_data = torch.empty_like(inputs, dtype=torch.uint8, device="cpu")

    # The maximum fp32/bf16 value is 2^128-eps. Divided by 448 (=1.75*2^8) it
    # becomes (1/1.75-eps)*2^120, with 0.5 < 1/1.75 < 1. Rounding this up gives
    # a scale of 2^120 and a corresponding value of 2^8.
    # We can't achieve scale == 254 (the max) because we encode amax / 448.
    max_scale = 247  # 2^120
    max_data = 0x78  # 256.0

    smallest_subnormal_hp = torch.tensor(
        [1], dtype=getattr(torch, f"uint{torch.finfo(input_dtype).bits}"), device=device
    ).view(input_dtype)
    if input_dtype == torch.bfloat16:
        smallest_subnormal_data = 0x08  # 2^-133/2^-127 = 2^-6
    else:
        smallest_subnormal_data = 0x00  # 2^-149/2^-127 -> 0

    satfinite_hp = 480.0  # In between fp8_max (448) and the next power of 2 (512)
    satfinite_scale = 128  # 2.0
    satfinite_data = 0x77  # 240.0

    for idx, name in enumerate(names):
        if name == "zero":
            inputs[idx].fill_(0.0)
            expected_scales[idx] = 0  # 2^-127
            expected_data[idx].fill_(0x00)

        elif name == "smallest_normal":
            # E8M0 byte 0 encodes 2^-127 rather than numerical zero. The smallest
            # FP32/BF16 normal therefore uses scale 2^-127 and quantizes to 2.0.
            inputs[idx].fill_(torch.finfo(input_dtype).tiny)  # 2^-126
            expected_scales[idx] = 0  # 2^-127
            expected_data[idx].fill_(0x40)  # 2.0

        elif name == "mixed_inf":
            # Inf determines the scale and invalidates the block. This is the
            # only case that distinguishes a no-saturation E8M0 conversion from
            # a saturating one: NOSAT maps Inf to 0xff (NaN), making the descale
            # and hence every output NaN, whereas SATFINITE would clamp it to
            # 0xfe (2^127). NaN itself maps to 0xff under both modes.
            inputs[idx].fill_(1.0)
            inputs[idx, 0] = float("inf")
            expected_scales[idx] = 255  # NaN
            expected_data[idx].fill_(0x7F)  # NaN

        elif name == "mixed_nan":
            # Any NaN invalidates the block.
            inputs[idx].fill_(1.0)
            inputs[idx, 0] = float("nan")
            expected_scales[idx] = 255  # NaN
            expected_data[idx].fill_(0x7F)  # NaN

        elif name == "positive_fp8_max":
            inputs[idx].fill_(448.0)
            expected_scales[idx] = 127  # 1.0
            expected_data[idx].fill_(0x7E)  # 448.0

        elif name == "negative_fp8_max":
            inputs[idx].fill_(-448.0)
            expected_scales[idx] = 127  # 1.0
            expected_data[idx].fill_(0xFE)  # -448.0

        elif name == "positive_dtype_max":
            inputs[idx].fill_(torch.finfo(input_dtype).max)
            expected_scales[idx] = max_scale
            expected_data[idx].fill_(max_data)

        elif name == "negative_dtype_max":
            inputs[idx].fill_(-torch.finfo(input_dtype).max)
            expected_scales[idx] = max_scale
            expected_data[idx].fill_(max_data | 0x80)

        elif name == "positive_smallest_subnormal":
            # The smallest BF16 subnormal reaches the FP8 minimum normal at scale
            # 2^-127. The smallest FP32 subnormal underflows to signed zero.
            inputs[idx] = smallest_subnormal_hp
            expected_scales[idx] = 0  # 2^-127
            expected_data[idx].fill_(smallest_subnormal_data)

        elif name == "negative_smallest_subnormal":
            inputs[idx] = -smallest_subnormal_hp
            expected_scales[idx] = 0  # 2^-127
            expected_data[idx].fill_(smallest_subnormal_data | 0x80)

        elif name == "mixed_positive_underflow":
            # With 1.0 determining the scale, the smallest subnormal underflows. The
            # negative case checks that FP8 conversion preserves the zero sign bit.
            inputs[idx].fill_(1.0)
            inputs[idx, 0] = smallest_subnormal_hp
            expected_scales[idx] = 119  # 2^-8
            expected_data[idx].fill_(0x78)  # 256
            expected_data[idx, 0] = 0x00  # 0.0

        elif name == "mixed_negative_underflow":
            inputs[idx].fill_(1.0)
            inputs[idx, 0] = -smallest_subnormal_hp
            expected_scales[idx] = 119  # 2^-8
            expected_data[idx].fill_(0x78)  # 256
            expected_data[idx, 0] = 0x80  # -0.0

        elif name == "mixed_saturation_and_underflow":
            # The largest value saturates while its finite peers underflow using the
            # same scale.
            inputs[idx].fill_(1.0)
            inputs[idx, 0] = torch.finfo(input_dtype).max
            expected_scales[idx] = max_scale
            expected_data[idx].fill_(0x00)
            expected_data[idx, 0] = max_data

        elif name == "rceil_premature_bf16_lowers_scale":
            # This guards against a regression where a fp32 -> bf16 -> ue8m0
            # chain would cause us to round the other way wrt fp32 -> ue8m0.
            # As 448 is 3.5*2^7, we choose 3.5 plus one FP32 ULP, because it
            # becomes exactly 3.5 when cast to BF16.
            boundary = (
                torch.tensor(0x40600001, dtype=torch.uint32, device=device)
                .view(torch.float32)
                .to(input_dtype)
            )
            inputs[idx].fill_(boundary)
            if input_dtype == torch.float32:
                expected_scales[idx] = 121  # (3.5+eps)/448 = 2^-7+eps -> 2^-6
                expected_data[idx].fill_(0x76)  # 224.0
            else:
                # If we round down instead of up, or if we used a bf16 input
                # where that 1 ULP eps was erased, the scale is one less.
                expected_scales[idx] = 120  # 2^-7
                expected_data[idx].fill_(0x7E)  # 448.0

        elif name == "fp32_just_below_512":
            # The largest FP32 below 512 rounds to 512 in BF16. RCEIL selects
            # scale 2 for both inputs.
            boundary = (
                torch.tensor(0x43FFFFFF, dtype=torch.uint32, device=device)
                .view(torch.float32)
                .to(input_dtype)
            )
            inputs[idx].fill_(boundary)
            expected_scales[idx] = 128  # 2^1
            expected_data[idx].fill_(0x78)  # 256.0

        elif name == "all_nan":
            inputs[idx].fill_(float("nan"))
            expected_scales[idx] = 255  # NaN
            expected_data[idx].fill_(0x7F)  # NaN

        elif name == "positive_satfinite":
            # RCEIL selects scale 2 and normalizes these values to +/-240.
            inputs[idx].fill_(satfinite_hp)
            expected_scales[idx] = satfinite_scale
            expected_data[idx].fill_(satfinite_data)

        elif name == "negative_satfinite":
            inputs[idx].fill_(-satfinite_hp)
            expected_scales[idx] = satfinite_scale
            expected_data[idx].fill_(satfinite_data | 0x80)

        elif name == "fp32_subnormals":
            # Migrated from torchao's test_to_mx_rceil ("fp32 denorm").
            # Varied FP32 subnormals stay nonzero at the minimum E8M0 scale.
            # fmt: off
            inputs[idx] = torch.tensor(
                [
                    6142315, 5096174, 3345704, 6178415,
                    5728750, 419002, 1716691, 4335089,
                    5785800, 6234845, 1697524, 33075,
                    3975816, 3714822, 5411407, 3040844,
                    7400945, 4474166, 7257182, 1273750,
                    5872176, 4694081, 2096530, 6273621,
                    67028, 7585260, 4532315, 4599275,
                    6133942, 4542483, 5992199, 6862780,
                ],
                dtype=torch.uint32,
            ).view(torch.float32).to(device)
            expected_scales[idx] = 0  # 2^-127
            expected_data[idx] = torch.tensor(
                [
                    60, 58, 53, 60, 59, 29, 45, 56,
                    59, 60, 45, 4, 55, 54, 58, 52,
                    62, 57, 62, 42, 59, 57, 48, 60,
                    8, 62, 57, 57, 60, 57, 59, 61,
                ],
                dtype=torch.uint8,
            )
            # fmt: on

        elif name == "bf16_subnormals":
            # Migrated from torchao's test_to_mx_rceil ("bf16 denorm").
            # Varied BF16 subnormals stay nonzero at the minimum E8M0 scale.
            # fmt: off
            inputs[idx] = torch.tensor(
                [
                    101, 3, 47, 54, 36, 19, 70, 79,
                    35, 95, 28, 120, 84, 94, 20, 92,
                    18, 42, 98, 58, 3, 26, 64, 86,
                    60, 86, 52, 23, 61, 70, 59, 74,
                ],
                dtype=torch.uint16,
            ).view(torch.bfloat16).to(device)
            expected_scales[idx] = 0  # 2^-127
            expected_data[idx] = torch.tensor(
                [
                    61, 20, 52, 54, 49, 42, 57, 58,
                    49, 60, 46, 63, 58, 60, 42, 60,
                    41, 50, 60, 54, 20, 45, 56, 59,
                    55, 59, 53, 44, 55, 57, 55, 57,
                ],
                dtype=torch.uint8,
            )
            # fmt: on

        elif name == "fp32_realistic_vals":
            # Migrated from torchao's test_to_mx_rceil ("fp32 normal").
            # Varied normal FP32 values exercise E4M3 rounding at scale 2^-8.
            # fmt: off
            inputs[idx] = torch.tensor(
                [
                    1037408064, 1058534842, 1053630662, 1063310394,
                    994704128, 1057245441, 1060663708, 1058053571,
                    1052395648, 1064831570, 1038427336, 1064777688,
                    1059248393, 1060959028, 1062878286, 1057799482,
                    1057854101, 1053562724, 1027482352, 1060498324,
                    1063238522, 1060472055, 1054346794, 1029092912,
                    1056687298, 1059146141, 1037992128, 1064097772,
                    1056522806, 1059255744, 1064364912, 1060606252,
                ],
                dtype=torch.uint32,
            ).view(torch.float32).to(device)
            expected_scales[idx] = 119  # 2^-8
            expected_data[idx] = torch.tensor(
                [
                    93, 113, 109, 118, 53, 112, 116, 113,
                    108, 120, 94, 119, 114, 116, 118, 113,
                    113, 109, 84, 115, 118, 115, 110, 85,
                    112, 114, 94, 119, 112, 114, 119, 115,
                ],
                dtype=torch.uint8,
            )
            # fmt: on

        elif name == "bf16_realistic_vals":
            # Migrated from torchao's test_to_mx_rceil ("bf16 normal").
            # Varied normal BF16 values exercise E4M3 rounding at scale 2^-8.
            # fmt: off
            inputs[idx] = torch.tensor(
                [
                    15752, 16143, 16182, 15896, 16195, 16186, 16048, 16223,
                    15988, 16231, 16140, 16088, 16032, 16240, 16228, 16133,
                    16210, 16024, 16248, 16187, 16050, 15696, 16060, 15956,
                    16131, 16251, 15896, 16014, 15808, 16024, 16159, 16186,
                ],
                dtype=torch.uint16,
            ).view(torch.bfloat16).to(device)
            expected_scales[idx] = 119  # 2^-8
            expected_data[idx] = torch.tensor(
                [
                    88, 113, 115, 98, 116, 116, 107, 118,
                    103, 118, 113, 110, 106, 119, 118, 112,
                    117, 106, 120, 116, 107, 85, 108, 101,
                    112, 120, 98, 105, 92, 106, 114, 116,
                ],
                dtype=torch.uint8,
            )
            # fmt: on

        else:
            raise RuntimeError(f"Bad name: {name}")

    expected_scales = expected_scales.reshape(-1, 1)
    return MXFP8SemanticCases(names, inputs, expected_data, expected_scales)


def assert_mxfp8_semantics(
    actual_data: torch.Tensor,
    actual_scales: torch.Tensor,
    cases: MXFP8SemanticCases,
) -> None:
    r"""Assert exact finite bytes and report all semantic case mismatches.

    E4M3FN has positive and negative NaN encodings (0x7f and 0xff). Their sign
    is not semantically meaningful and can differ between hardware conversion
    instructions, so treat the two encodings as equivalent.
    """
    actual_data = actual_data.view(torch.uint8).cpu()
    actual_scales = actual_scales.view(torch.uint8).cpu()
    if actual_data.shape != cases.expected_data.shape:
        raise AssertionError(
            f"data shape {actual_data.shape} != {cases.expected_data.shape}"
        )
    if actual_scales.shape != cases.expected_scales.shape:
        raise AssertionError(
            f"scale shape {actual_scales.shape} != {cases.expected_scales.shape}"
        )

    errors = []
    for case_idx, case_name in enumerate(cases.names):
        actual_scale = actual_scales[case_idx]
        expected_scale = cases.expected_scales[case_idx]
        if not torch.equal(actual_scale, expected_scale):
            errors.append(
                f"scale mismatch for {case_name}: "
                f"actual={actual_scale.tolist()}, expected={expected_scale.tolist()}"
            )

        actual = actual_data[case_idx]
        expected = cases.expected_data[case_idx]
        actual_normalized = torch.where(actual == 0xFF, 0x7F, actual)
        if not torch.equal(actual_normalized, expected):
            errors.append(
                f"data mismatch for {case_name}: "
                f"actual={actual.tolist()}, expected={expected.tolist()}"
            )

    if errors:
        raise AssertionError("\n" + "\n".join(errors))


def make_f32_to_e8m0_rceil_cases(*, device):
    r"""Create exact FP32 bit patterns around E8M0 RCEIL boundaries."""
    # copied from
    # https://github.com/pytorch/ao/blob/3972ed015091f659418dedf12edb980a8ca56b53/torchao/testing/_mxfp8_test_utils.py#L274
    values_and_expected_scales = [
        (0x00000000, 0),  # zero
        (0x00000001, 0),  # smallest FP32 subnormal
        (0x00400000, 0),  # 2^-127, exactly E8M0 byte 0
        (0x00400001, 1),  # just above 2^-127
        (0x007FFFFF, 1),  # largest FP32 subnormal
        (0x00800000, 1),  # 2^-126, exactly E8M0 byte 1
        (0x1F800000, 63),  # 2^-64, exactly E8M0 byte 63
        (0x1F800001, 64),  # just above 2^-64
        (0x3F7FFFFF, 127),  # just below 1.0
        (0x3F800000, 127),  # 1.0
        (0x3F800001, 128),  # just above 1.0
        (0x5F800000, 191),  # 2^64, exactly E8M0 byte 191
        (0x5F800001, 192),  # just above 2^64
        (0x7F000000, 254),  # 2^127, exactly E8M0 byte 254
        (0x7F000001, 255),  # just above 2^127, overflows to E8M0 NaN
        (0x7F7FFFFF, 255),  # largest finite FP32 value
        (0x7F800000, 255),  # infinity
        (0x7FC00000, 255),  # NaN
    ]
    values = torch.tensor(
        [v for v, _ in values_and_expected_scales],
        dtype=torch.int32,
        device=device,
    ).view(torch.float32)
    expected = torch.tensor(
        [s for _, s in values_and_expected_scales],
        dtype=torch.uint8,
    )
    return values, expected
