// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>

#include "ck/ck.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

using namespace ck;

void traverse_using_space_filling_curve_linear();
void traverse_using_space_filling_curve_snakecurved();

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    traverse_using_space_filling_curve_linear();
    traverse_using_space_filling_curve_snakecurved();

    return 0;
}

void traverse_using_space_filling_curve_linear()
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    using TensorLengths    = Sequence<3, 2, 2>;
    using DimAccessOrder   = Sequence<2, 0, 1>;
    using ScalarsPerAccess = Sequence<1, 1, 1>;
    using SpaceFillingCurve =
        SpaceFillingCurve<TensorLengths, DimAccessOrder, ScalarsPerAccess, false>;

    constexpr auto expected = make_tuple(make_tuple(0, 0, 0),
                                         make_tuple(0, 1, 0),
                                         make_tuple(1, 0, 0),
                                         make_tuple(1, 1, 0),
                                         make_tuple(2, 0, 0),
                                         make_tuple(2, 1, 0),
                                         make_tuple(0, 0, 1),
                                         make_tuple(0, 1, 1),
                                         make_tuple(1, 0, 1),
                                         make_tuple(1, 1, 1),
                                         make_tuple(2, 0, 1),
                                         make_tuple(2, 1, 1));

    constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();

    static_assert(num_access == reduce_on_sequence(TensorLengths{} / ScalarsPerAccess{},
                                                   math::multiplies{},
                                                   Number<1>{}));

    static_for<1, num_access, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto backward_step = SpaceFillingCurve::GetBackwardStep(i);
        constexpr auto expected_step = expected[i - I1] - expected[i];
        static_assert(backward_step[I0] == expected_step[I0]);
        static_assert(backward_step[I1] == expected_step[I1]);
        static_assert(backward_step[I2] == expected_step[I2]);
    });

    static_for<0, num_access - 1, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto forward_step  = SpaceFillingCurve::GetForwardStep(i);
        constexpr auto expected_step = expected[i + I1] - expected[i];
        static_assert(forward_step[I0] == expected_step[I0]);
        static_assert(forward_step[I1] == expected_step[I1]);
        static_assert(forward_step[I2] == expected_step[I2]);
    });
}

void traverse_using_space_filling_curve_snakecurved()
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};

    using TensorLengths    = Sequence<16, 10, 9>;
    using DimAccessOrder   = Sequence<2, 0, 1>;
    using ScalarsPerAccess = Sequence<4, 2, 3>;
    using SpaceFillingCurve =
        SpaceFillingCurve<TensorLengths, DimAccessOrder, ScalarsPerAccess, true>;

    constexpr auto expected = make_tuple(make_tuple(0, 0, 0),
                                         make_tuple(0, 2, 0),
                                         make_tuple(0, 4, 0),
                                         make_tuple(0, 6, 0),
                                         make_tuple(0, 8, 0),
                                         make_tuple(4, 8, 0),
                                         make_tuple(4, 6, 0),
                                         make_tuple(4, 4, 0),
                                         make_tuple(4, 2, 0),
                                         make_tuple(4, 0, 0),
                                         make_tuple(8, 0, 0),
                                         make_tuple(8, 2, 0),
                                         make_tuple(8, 4, 0),
                                         make_tuple(8, 6, 0),
                                         make_tuple(8, 8, 0),
                                         make_tuple(12, 8, 0),
                                         make_tuple(12, 6, 0),
                                         make_tuple(12, 4, 0),
                                         make_tuple(12, 2, 0),
                                         make_tuple(12, 0, 0),
                                         make_tuple(12, 0, 3),
                                         make_tuple(12, 2, 3),
                                         make_tuple(12, 4, 3),
                                         make_tuple(12, 6, 3),
                                         make_tuple(12, 8, 3),
                                         make_tuple(8, 8, 3),
                                         make_tuple(8, 6, 3),
                                         make_tuple(8, 4, 3),
                                         make_tuple(8, 2, 3),
                                         make_tuple(8, 0, 3),
                                         make_tuple(4, 0, 3),
                                         make_tuple(4, 2, 3),
                                         make_tuple(4, 4, 3),
                                         make_tuple(4, 6, 3),
                                         make_tuple(4, 8, 3),
                                         make_tuple(0, 8, 3),
                                         make_tuple(0, 6, 3),
                                         make_tuple(0, 4, 3),
                                         make_tuple(0, 2, 3),
                                         make_tuple(0, 0, 3),
                                         make_tuple(0, 0, 6),
                                         make_tuple(0, 2, 6),
                                         make_tuple(0, 4, 6),
                                         make_tuple(0, 6, 6),
                                         make_tuple(0, 8, 6),
                                         make_tuple(4, 8, 6),
                                         make_tuple(4, 6, 6),
                                         make_tuple(4, 4, 6),
                                         make_tuple(4, 2, 6),
                                         make_tuple(4, 0, 6),
                                         make_tuple(8, 0, 6),
                                         make_tuple(8, 2, 6),
                                         make_tuple(8, 4, 6),
                                         make_tuple(8, 6, 6),
                                         make_tuple(8, 8, 6),
                                         make_tuple(12, 8, 6),
                                         make_tuple(12, 6, 6),
                                         make_tuple(12, 4, 6),
                                         make_tuple(12, 2, 6),
                                         make_tuple(12, 0, 6));

    constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();

    static_assert(num_access == reduce_on_sequence(TensorLengths{} / ScalarsPerAccess{},
                                                   math::multiplies{},
                                                   Number<1>{}));

    static_for<1, num_access, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto backward_step = SpaceFillingCurve::GetBackwardStep(i);
        constexpr auto expected_step = expected[i - I1] - expected[i];
        static_assert(backward_step[I0] == expected_step[I0]);
        static_assert(backward_step[I1] == expected_step[I1]);
        static_assert(backward_step[I2] == expected_step[I2]);
    });

    static_for<0, num_access - 1, 1>{}([&](auto i) {
        constexpr auto idx_curr = SpaceFillingCurve::GetIndex(i);

        static_assert(idx_curr[I0] == expected[i][I0]);
        static_assert(idx_curr[I1] == expected[i][I1]);
        static_assert(idx_curr[I2] == expected[i][I2]);

        constexpr auto forward_step  = SpaceFillingCurve::GetForwardStep(i);
        constexpr auto expected_step = expected[i + I1] - expected[i];
        static_assert(forward_step[I0] == expected_step[I0]);
        static_assert(forward_step[I1] == expected_step[I1]);
        static_assert(forward_step[I2] == expected_step[I2]);
    });
}
