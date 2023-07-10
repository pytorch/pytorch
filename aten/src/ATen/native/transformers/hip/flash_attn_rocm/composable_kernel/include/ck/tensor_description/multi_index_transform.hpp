// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/multi_index.hpp"

namespace ck {

template <typename LowLength>
struct PassThrough
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr PassThrough() = default;

    __host__ __device__ constexpr PassThrough(const LowLength& low_length)
        : up_lengths_{make_tuple(low_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("PassThrough, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLength,
          typename LeftPadLength,
          typename RightPadLength,
          bool SkipIsValidCheck = false>
struct Pad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;
    RightPadLength right_pad_length_;

    __host__ __device__ constexpr Pad() = default;

    __host__ __device__ constexpr Pad(const LowLength& low_length,
                                      const LeftPadLength& left_pad_length,
                                      const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length + right_pad_length)},
          left_pad_length_{left_pad_length},
          right_pad_length_{right_pad_length}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck ||
               ((idx_up[Number<0>{}] >= left_pad_length_) &&
                (idx_up[Number<0>{}] < up_lengths_[Number<0>{}] - right_pad_length_));
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPadLength>::value &&
               is_known_at_compile_time<RightPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Pad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("left_pad_length %d", index_t{left_pad_length_});
        printf("right_pad_length %d", index_t{right_pad_length_});
        printf("}");
    }
};

template <typename LowLength, typename LeftPadLength, bool SkipIsValidCheck = false>
struct LeftPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + LeftPadLength{}));

    UpLengths up_lengths_;
    LeftPadLength left_pad_length_;

    __host__ __device__ constexpr LeftPad() = default;

    __host__ __device__ constexpr LeftPad(const LowLength& low_length,
                                          const LeftPadLength& left_pad_length)
        : up_lengths_{make_tuple(low_length + left_pad_length)}, left_pad_length_{left_pad_length}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] - left_pad_length_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] >= left_pad_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LeftPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("LeftPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("left_pad_length_ %d", index_t{left_pad_length_});
        printf("}");
    }
};

template <typename LowLength, typename RightPadLength, bool SkipIsValidCheck = false>
struct RightPad
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(LowLength{} + RightPadLength{}));

    UpLengths up_lengths_;
    LowLength low_length_;
    RightPadLength right_pad_length_;

    __host__ __device__ constexpr RightPad() = default;

    __host__ __device__ constexpr RightPad(const LowLength& low_length,
                                           const RightPadLength& right_pad_length)
        : up_lengths_{make_tuple(low_length + right_pad_length)},
          low_length_{low_length},
          right_pad_length_{right_pad_length}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                                  const UpIdx& idx_up)
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return SkipIsValidCheck;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& idx_up) const
    {
        return SkipIsValidCheck || (idx_up[Number<0>{}] < low_length_);
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<LowLength>::value &&
               is_known_at_compile_time<RightPadLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("RightPad, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("low_length_ %d", index_t{low_length_});
        printf("left_pad_length_ %d", index_t{right_pad_length_});
        printf("}");
    }
};

// idx_low = coefficients[0, ...nDimUp-1] * idx_up[0, ...nDimUp-1]
// UpLengths and Coefficients can be either of the followings:
//   1) Tuple of index_t, which is known at run-time, or
//   2) Tuple of Number, which is known at compile-time, or
//   3) Tuple of mixture of index_t and Number, which is known partially at run-time and partially
//   at compile-time
template <typename UpLengths,
          typename Coefficients,
          typename enable_if<UpLengths::Size() == Coefficients::Size(), bool>::type = false>
struct Embed
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    UpLengths up_lengths_;
    Coefficients coefficients_;

    __host__ __device__ constexpr Embed() = default;

    __host__ __device__ constexpr Embed(const UpLengths& up_lengths,
                                        const Coefficients& coefficients)
        : up_lengths_{up_lengths}, coefficients_{coefficients}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}([&idx_low, &idx_up, this](auto i) {
            idx_low(Number<0>{}) += idx_up[i] * this->coefficients_[i];
        });
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == NDimUp &&
                          LowIdx::Size() == 1 && UpIdx::Size() == NDimUp,
                      "wrong! inconsistent # of dimension");

        idx_diff_low(Number<0>{}) = 0;

        static_for<0, NDimUp, 1>{}(
            [&](auto i) { idx_diff_low(Number<0>{}) += idx_diff_up[i] * coefficients_[i]; });

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<Coefficients>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Embed, ");
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("coefficients_ ");
        print_multi_index(coefficients_);
        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses regular to do lowering of
// multi-index and use carry-and-borrow check to do lowering of multi-index delta
template <typename LowLengths>
struct Merge_v1_carry_check
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v1_carry_check() = default;

    __host__ __device__ constexpr Merge_v1_carry_check(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        // normal division
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_1a(LowIdxDiff& idx_diff_low,
                                                 const UpIdxDiff& idx_diff_up,
                                                 LowIdx& idx_low,
                                                 const UpIdx& /* idx_up_new */,
                                                 Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        LowerIndex idx_low_length_minus_idx_diff_low_const;
        LowerIndex idx_low_length_plus_idx_diff_low_const;

#if !CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) = low_lengths_[i] - idx_diff_low_const[i];

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] - idx_diff_low_const[i]);

            idx_low_length_plus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] + idx_diff_low_const[i]);
        });
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
        else if constexpr(Hack == 2)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] - borrow;

                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) -= borrow;

                borrow = do_borrow ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] - borrow;

            idx_low += idx_diff_low;
        }
        else
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry  = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];
                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];
                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
                carry = do_borrow ? -1 : carry;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_1b(LowIdxDiff& idx_diff_low,
                                                 const UpIdxDiff& idx_diff_up,
                                                 LowIdx& idx_low,
                                                 const UpIdx& /* idx_up_new */,
                                                 Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        // computed at
        //   run-time each time this function is called, and can be very expensive.
        LowerIndex idx_diff_low_const;
        LowerIndex idx_low_length_minus_idx_diff_low_const;
        LowerIndex idx_low_length_plus_idx_diff_low_const;

#if !CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) = low_lengths_[i] - idx_diff_low_const[i];

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);

        static_for<0, NDimLow, 1>{}([&](auto i) {
            idx_low_length_minus_idx_diff_low_const(i) =
                __builtin_amdgcn_readfirstlane(low_lengths_[i] - idx_diff_low_const[i]);

            idx_low_length_plus_idx_diff_low_const(i) = low_lengths_[i] + idx_diff_low_const[i];
        });
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
        else if constexpr(Hack == 2)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t negative_idx_low_tmp = borrow - idx_low[i];

                bool do_borrow = negative_idx_low_tmp > idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low_const[i];

                idx_diff_low(i) -= borrow;

                borrow = do_borrow ? 1 : 0;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] - borrow;

            idx_low += idx_diff_low;
        }
        else
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            index_t carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                index_t idx_low_tmp = idx_low[i] + carry;

                bool do_carry  = idx_low_tmp >= idx_low_length_minus_idx_diff_low_const[i];
                bool do_borrow = idx_low_tmp < -idx_diff_low_const[i];

                idx_diff_low(i) =
                    do_carry ? -idx_low_length_minus_idx_diff_low_const[i] : idx_diff_low_const[i];
                idx_diff_low(i) =
                    do_borrow ? idx_low_length_plus_idx_diff_low_const[i] : idx_diff_low[i];

                idx_diff_low(i) += carry;

                carry = do_carry ? 1 : 0;
                carry = do_borrow ? -1 : carry;
            });

            idx_diff_low(Number<0>{}) = idx_diff_low_const[Number<0>{}] + carry;

            idx_low += idx_diff_low;
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex_2(LowIdxDiff& idx_diff_low,
                                                const UpIdxDiff& idx_diff_up,
                                                LowIdx& idx_low,
                                                const UpIdx& /* idx_up_new */,
                                                Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        // CalculateLowerIndex(idx_diff_low_const) has multiple integer divisions.
        // However,
        //   1) If idx_diff_up is known at compile-time, then idx_diff_low_const
        //   can be calculated at compile-time.
        //   2) If idx_diff_up is not known at compile-time, but its value
        //   doesn't change during the whole kernel execution, then
        //   idx_diff_low_const also
        //   doesn't change during the whole kernel execution. Compiler generated
        //   ISA should
        //   only caclculate idx_diff_low_const once and save it durinng the whole
        //   kernel execution
        // If neither 1) nor 2) is satisfied, then the calculation will also be
        //   computed at run-time each time this function is called, and can be
        //   very expensive.
        LowerIndex idx_diff_low_const;

#if !CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = tmp / low_lengths_scan_[i];
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = tmp;
#else
        // Hack: this force result into SGPR. Need to make sure the result is thread invariant
        index_t tmp = idx_diff_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_diff_low_const(i) = __builtin_amdgcn_readfirstlane(tmp / low_lengths_scan_[i]);
            tmp -= idx_diff_low_const[i] * low_lengths_scan_[i];
        });

        idx_diff_low_const(Number<NDimLow - 1>{}) = __builtin_amdgcn_readfirstlane(tmp);
#endif

        if constexpr(Hack == 1)
        {
            // do carry check on each low dimension in reversed order
            // do not need to check the first dimension
            bool do_carry = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                idx_diff_low(i) = idx_diff_low_const[i] + do_carry;

                index_t idx_low_tmp = idx_low[i] + idx_diff_low[i];

                do_carry = idx_low_tmp >= low_lengths_[i];

#if 0
                // TODO: use exec-mask inline asm, which use 1 VALU
                if(do_carry)
                {
                    idx_diff_low(i) -= low_lengths_[i];
                }
#elif 1
                // this use 2 VALU
                idx_diff_low(i) = do_carry ? idx_diff_low[i] - low_lengths_[i] : idx_diff_low[i];
#elif 1
                // this use 2 VALU
                index_t idx_diff_low_tmp = idx_diff_low[i] - low_lengths_[i];
                idx_diff_low(i)          = do_carry ? idx_diff_low_tmp : idx_diff_low[i];
#endif

                idx_low(i) += idx_diff_low[i];
            });

            constexpr auto I0 = Number<0>{};

            idx_diff_low(I0) = idx_diff_low_const[I0] + do_carry;

            idx_low(I0) += idx_diff_low[I0];
        }
        else if constexpr(Hack == 2)
        {
            // do borrow check on each low dimension in reversed order
            // do not need to check the first dimension
            bool do_borrow = 0;

            static_for<NDimLow - 1, 0, -1>{}([&](auto i) {
                idx_diff_low(i) = idx_diff_low_const[i] - do_borrow;

                index_t idx_low_tmp = idx_low[i] + idx_diff_low[i];

                do_borrow = idx_low_tmp < 0;

#if 0
                // TODO: use exec-mask inline asm
                if(do_borrow)
                {
                    idx_diff_low(i) += low_lengths_[i];
                }
#elif 1
                idx_diff_low(i) = do_borrow ? idx_diff_low[i] + low_lengths_[i] : idx_diff_low[i];
#elif 1
                index_t idx_diff_low_tmp = idx_diff_low[i] + low_lengths_[i];
                idx_diff_low(i)          = do_borrow ? idx_diff_low_tmp : idx_diff_low[i];
#endif

                idx_low(i) += idx_diff_low[i];
            });

            constexpr auto I0 = Number<0>{};

            idx_diff_low(I0) = idx_diff_low_const[I0] - do_borrow;

            idx_low(I0) += idx_diff_low[I0];
        }
        else
        {
            // not implemented
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
#if 1
        UpdateLowerIndex_1a(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#elif 0
        UpdateLowerIndex_1b(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#else
        UpdateLowerIndex_2(idx_diff_low, idx_diff_up, idx_low, idx_up_new, Number<Hack>{});
#endif
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v1_carry_check, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan_ ");
        print_multi_index(low_lengths_scan_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLengths>
struct lambda_merge_generate_MagicDivision_calculate_magic_multiplier
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return MagicDivision::CalculateMagicMultiplier(LowLengths{}[i]);
    }
};

template <typename LowLengths>
struct lambda_merge_generate_MagicDivision_calculate_magic_shift
{
    template <index_t I>
    __host__ __device__ constexpr auto operator()(Number<I> i) const
    {
        return MagicDivision::CalculateMagicShift(LowLengths{}[i]);
    }
};

// Implementation of "Merge" transformation primitive that uses magic-number-division to do lowering
// of both multi-index and delta of multi-index
// Caution:
//   1. The magic number division implementation being used would produce correct result if the
//   dividended is uint32_t and its value is with in 31-bit value range of uint32_t.
//   2. The magic number division for int32_t dividened has not been implemented, the int32_t
//   dividend would be bit-wise interpreted as uint32_t and magic number division implementation for
//   uint32_t is then used.
//   3. For Merge primitive, upper-index is the dividend.
//   4. When upper-index is uint32_t, its value need to be within 31-bit range.
//   5. When upper-index is int32_t type (when index_t is int32_t), its value need to be
//   non-negative.
template <typename LowLengths>
struct Merge_v2_magic_division
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    using LowLengthsMagicDivisorMultipiler = decltype(
        generate_tuple(lambda_merge_generate_MagicDivision_calculate_magic_multiplier<LowLengths>{},
                       Number<NDimLow>{}));

    using LowLengthsMagicDivisorShift = decltype(
        generate_tuple(lambda_merge_generate_MagicDivision_calculate_magic_shift<LowLengths>{},
                       Number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsMagicDivisorMultipiler low_lengths_magic_divisor_multiplier_;
    LowLengthsMagicDivisorShift low_lengths_magic_divisor_shift_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v2_magic_division() = default;

    __host__ __device__ constexpr Merge_v2_magic_division(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_magic_divisor_multiplier_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicMultiplier(low_lengths[i]); },
              Number<NDimLow>{})},
          low_lengths_magic_divisor_shift_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicShift(low_lengths[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_magic_divisor_multiplier_[i],
                                               this->low_lengths_magic_divisor_shift_[i]);
            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;
        });

        idx_low(Number<0>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up_new[Number<0>{}];

        static_for<NDimLow - 1, 0, -1>{}([&, this](auto i) {
            index_t tmp2 =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_magic_divisor_multiplier_[i],
                                               this->low_lengths_magic_divisor_shift_[i]);

            index_t idx_low_old = idx_low[i];

            idx_low(i) = tmp - tmp2 * this->low_lengths_[i];
            tmp        = tmp2;

            idx_diff_low(i) = idx_low[i] - idx_low_old;
        });

        idx_diff_low(Number<0>{}) = tmp - idx_low(Number<0>{});

        idx_low(Number<0>{}) = tmp;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisorMultipiler>::value &&
               is_known_at_compile_time<LowLengthsMagicDivisorShift>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v2_magic_division, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_magic_divisor_multiplier_ ");
        print_multi_index(low_lengths_magic_divisor_multiplier_);
        printf("low_lengths_magic_divisor_shift_ ");
        print_multi_index(low_lengths_magic_divisor_shift_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses magic-number-division to do lowering
// of both multi-index and delta of multi-index
// Caution:
//   1. The magic number division implementation being used would produce correct result if the
//   dividended is uint32_t and its value is with in 31-bit value range of uint32_t.
//   2. The magic number division for int32_t dividened has not been implemented, the int32_t
//   dividend would be bit-wise interpreted as uint32_t and magic number division implementation for
//   uint32_t is then used.
//   3. For Merge primitive, upper-index is the dividend.
//   4. When upper-index is uint32_t, its value need to be within 31-bit range.
//   5. When upper-index is int32_t type (when index_t is int32_t), its value need to be
//   non-negative.
template <typename LowLengths>
struct Merge_v2r2_magic_division
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    using LowLengthsScanMagicDivisorMultipiler = decltype(generate_tuple(
        lambda_merge_generate_MagicDivision_calculate_magic_multiplier<LowLengthsScan>{},
        Number<NDimLow>{}));

    using LowLengthsScanMagicDivisorShift = decltype(
        generate_tuple(lambda_merge_generate_MagicDivision_calculate_magic_shift<LowLengthsScan>{},
                       Number<NDimLow>{}));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    LowLengthsScanMagicDivisorMultipiler low_lengths_scan_magic_divisor_multiplier_;
    LowLengthsScanMagicDivisorShift low_lengths_scan_magic_divisor_shift_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v2r2_magic_division() = default;

    __host__ __device__ constexpr Merge_v2r2_magic_division(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          low_lengths_scan_magic_divisor_multiplier_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicMultiplier(low_lengths_scan_[i]); },
              Number<NDimLow>{})},
          low_lengths_scan_magic_divisor_shift_{generate_tuple(
              [&](auto i) { return MagicDivision::CalculateMagicShift(low_lengths_scan_[i]); },
              Number<NDimLow>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&, this](auto i) {
            idx_low(i) =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_scan_magic_divisor_multiplier_[i],
                                               this->low_lengths_scan_magic_divisor_shift_[i]);

            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up_new[Number<0>{}];

        static_for<0, NDimLow - 1, 1>{}([&, this](auto i) {
            index_t idx_low_old = idx_low[i];

            idx_low(i) =
                MagicDivision::DoMagicDivision(tmp,
                                               this->low_lengths_scan_magic_divisor_multiplier_[i],
                                               this->low_lengths_scan_magic_divisor_shift_[i]);

            idx_diff_low(i) = idx_low[i] - idx_low_old;

            tmp -= idx_low[i] * this->low_lengths_scan_[i];
        });

        idx_diff_low(Number<NDimLow - 1>{}) = tmp - idx_low[Number<NDimLow - 1>{}];

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScanMagicDivisorMultipiler>::value &&
               is_known_at_compile_time<LowLengthsScanMagicDivisorShift>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v2r2_magic_division, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan ");
        print_multi_index(low_lengths_scan_);
        printf("low_lengths_scan_magic_divisor_multiplier_ ");
        print_multi_index(low_lengths_scan_magic_divisor_multiplier_);
        printf("low_lengths_scan_magic_divisor_shift_ ");
        print_multi_index(low_lengths_scan_magic_divisor_shift_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

// Implementation of "Merge" transformation primitive that uses division and mod. It is supposed to
// be used for low_lengths that are known at compile time and are power of 2, otherwise performance
// will be very bad
template <typename LowLengths>
struct Merge_v3_division_mod
{
    static constexpr index_t NDimLow = LowLengths::Size();

    using LowerIndex = MultiIndex<NDimLow>;
    using UpperIndex = MultiIndex<1>;

    using LowLengthsScan =
        decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));

    using UpLengths =
        decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));

    LowLengths low_lengths_;
    LowLengthsScan low_lengths_scan_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Merge_v3_division_mod() = default;

    __host__ __device__ constexpr Merge_v3_division_mod(const LowLengths& low_lengths)
        : low_lengths_{low_lengths},
          low_lengths_scan_{
              container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
          up_lengths_{make_tuple(container_reduce(low_lengths, math::multiplies{}, Number<1>{}))}
    {
        static_assert(LowerIndex::Size() == NDimLow, "wrong!");
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return NDimLow; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        index_t tmp = idx_up[Number<0>{}];

        // division and mod
        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            idx_low(i) = tmp / this->low_lengths_scan_[i];
            tmp %= this->low_lengths_scan_[i];
        });

        idx_low(Number<NDimLow - 1>{}) = tmp;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff&,
                                              LowIdx& idx_low,
                                              const UpIdx& idx_up_new,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == NDimLow && UpIdxDiff::Size() == 1 &&
                          LowIdx::Size() == NDimLow && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0   = Number<0>{};
        constexpr auto INm1 = Number<NDimLow - 1>{};

        index_t tmp = idx_up_new[I0];

        static_for<0, NDimLow - 1, 1>{}([&](auto i) {
            const index_t tmp2 = idx_low[i];
            idx_low(i)         = tmp / this->low_lengths_scan_[i];
            idx_diff_low(i)    = idx_low[i] - tmp2;
            tmp %= this->low_lengths_scan_[i];
        });

        const index_t tmp2 = idx_low[INm1];
        idx_low(INm1)      = tmp;
        idx_diff_low(INm1) = idx_low[INm1] - tmp2;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowLengths>::value &&
               is_known_at_compile_time<LowLengthsScan>::value &&
               is_known_at_compile_time<UpLengths>::value;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Merge_v3_direct_division_mod, ");
        printf("low_lengths_ ");
        print_multi_index(low_lengths_);
        printf("low_lengths_scan_ ");
        print_multi_index(low_lengths_scan_);
        printf("up_lengths_ ");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename UpLengths, bool Use24BitIntegerCalculation>
struct UnMerge
{
    static constexpr index_t NDimUp = UpLengths::Size();

    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<NDimUp>;

    using UpLengthsScan =
        decltype(container_reverse_exclusive_scan(UpLengths{}, math::multiplies{}, Number<1>{}));

    UpLengths up_lengths_;
    UpLengthsScan up_lengths_scan_;

    __host__ __device__ constexpr UnMerge() = default;

    __host__ __device__ constexpr UnMerge(const UpLengths& up_lengths)
        : up_lengths_{up_lengths},
          up_lengths_scan_{
              container_reverse_exclusive_scan(up_lengths, math::multiplies{}, Number<1>{})}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return NDimUp; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        if constexpr(!Use24BitIntegerCalculation)
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}(
                [&](auto i) { idx_low(Number<0>{}) += idx_up[i] * up_lengths_scan_[i]; });
        }
        else
        {
            idx_low(Number<0>{}) = idx_up[Number<NDimUp - 1>{}];

            static_for<0, NDimUp - 1, 1>{}([&](auto i) {
                idx_low(Number<0>{}) =
                    (0x00ffffff & idx_low[Number<0>{}]) +
                    (0x00ffffff & idx_up[i]) * (0x00ffffff & up_lengths_scan_[i]);
            });
        }
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&,
                                              Number<Hack>) const
    {
        CalculateLowerIndex(idx_diff_low, idx_diff_up);

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<UpLengthsScan>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("UnMerge, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("up_lengths_scan_");
        print_multi_index(up_lengths_scan_);
        printf("}");
    }
};

template <typename LowerIndex>
struct Freeze
{
    LowerIndex low_idx_;

    __host__ __device__ constexpr Freeze() = default;

    __host__ __device__ constexpr Freeze(const LowerIndex& low_idx) : low_idx_{low_idx} {}

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 0; }

    __host__ __device__ static constexpr auto GetUpperLengths() { return Tuple<>{}; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& /* idx_up */) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 0,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = low_idx_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& /* idx_diff_up */,
                                                     LowIdx& /* idx_low */,
                                                     const UpIdx& /* idx_up_new */,
                                                     Number<Hack>)
    {
        idx_diff_low(Number<0>{}) = 0;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<LowerIndex>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Freeze");
        printf("low_idx_ %d", index_t{low_idx_});
    }
};

// Insert a dangling upper dimension without lower dimension
template <typename UpperLength>
struct Insert
{
    using UpLengths = decltype(make_tuple(UpperLength{}));

    UpLengths up_lengths_;

    __host__ __device__ constexpr Insert() = default;

    __host__ __device__ constexpr Insert(const UpperLength& up_length)
        : up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 0; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr auto GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx&, const UpIdx&) const
    {
        static_assert(LowIdx::Size() == 0 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void
    UpdateLowerIndex(LowIdxDiff&, const UpIdxDiff&, LowIdx&, const UpIdx&, Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 0 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 0 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpperLength>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("Insert");
        print_multi_index(up_lengths_);
    }
};

template <typename VectorSize, typename UpLength>
struct Vectorize
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(UpLength{}));

    UpLengths up_lengths_;
    VectorSize vector_size_;

    __host__ __device__ constexpr Vectorize() = default;

    __host__ __device__ constexpr Vectorize(const VectorSize& vector_size,
                                            const UpLength& up_length)
        : vector_size_{vector_size}, up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = vector_size_ * idx_up[Number<0>{}];
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx&,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = vector_size_ * idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Vectorize, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};

template <typename LowLength, typename SliceBegin, typename SliceEnd>
struct Slice
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;

    using UpLengths = decltype(make_tuple(SliceEnd{} - SliceBegin{}));

    UpLengths up_lengths_;
    SliceBegin slice_begin_;
    SliceEnd slice_end_;

    __host__ __device__ constexpr Slice() = default;

    __host__ __device__ constexpr Slice(const LowLength&,
                                        const SliceBegin& slice_begin,
                                        const SliceEnd& slice_end)
        : up_lengths_{make_tuple(slice_end - slice_begin)},
          slice_begin_{slice_begin},
          slice_end_{slice_end}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] + slice_begin_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ static void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                                     const UpIdxDiff& idx_diff_up,
                                                     LowIdx& idx_low,
                                                     const UpIdx&,
                                                     Number<Hack>)
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        idx_diff_low(I0) = idx_diff_up[I0];

        idx_low += idx_diff_low;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return true; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ constexpr bool IsValidUpperIndexMappedToValidLowerIndex(const UpIdx&) const
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value &&
               is_known_at_compile_time<SliceBegin>::value &&
               is_known_at_compile_time<SliceEnd>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Slice, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("slice_begin_ %d", index_t{slice_begin_});
        printf("slice_end %d", index_t{slice_end_});
        printf("}");
    }
};

/*
 * \brief lower_idx = upper_idx % modulus.
 * TODO: Need an improved implementation since the modulo operation is expensive.
 */
template <typename Modulus, typename UpLength>
struct Modulo
{
    using LowerIndex = MultiIndex<1>;
    using UpperIndex = MultiIndex<1>;
    using UpLengths  = decltype(make_tuple(UpLength{}));

    Modulus modulus_;
    UpLengths up_lengths_;

    __host__ __device__ constexpr Modulo() = default;

    __host__ __device__ constexpr Modulo(const Modulus& modulus, const UpLength& up_length)
        : modulus_{modulus}, up_lengths_{make_tuple(up_length)}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfLowerDimension() { return 1; }

    __host__ __device__ static constexpr index_t GetNumOfUpperDimension() { return 1; }

    __host__ __device__ constexpr const auto& GetUpperLengths() const { return up_lengths_; }

    template <typename LowIdx, typename UpIdx>
    __host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
                                                           const UpIdx& idx_up) const
    {
        static_assert(LowIdx::Size() == 1 && UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        idx_low(Number<0>{}) = idx_up[Number<0>{}] % modulus_;
    }

    template <typename LowIdxDiff,
              typename UpIdxDiff,
              typename LowIdx,
              typename UpIdx,
              index_t Hack>
    __host__ __device__ void UpdateLowerIndex(LowIdxDiff& idx_diff_low,
                                              const UpIdxDiff& idx_diff_up,
                                              LowIdx& idx_low,
                                              const UpIdx& up_idx,
                                              Number<Hack>) const
    {
        static_assert(LowIdxDiff::Size() == 1 && UpIdxDiff::Size() == 1 && LowIdx::Size() == 1 &&
                          UpIdx::Size() == 1,
                      "wrong! inconsistent # of dimension");

        constexpr auto I0 = Number<0>{};

        const auto idx_low_old = idx_low;
        idx_low(I0)            = (up_idx(I0) + idx_diff_up(I0)) % modulus_;
        idx_diff_low(I0)       = idx_low - idx_low_old;
    }

    __host__ __device__ static constexpr bool IsLinearTransform() { return false; }

    __host__ __device__ static constexpr bool IsValidUpperIndexAlwaysMappedToValidLowerIndex()
    {
        return true;
    }

    template <typename UpIdx>
    __host__ __device__ static constexpr bool
    IsValidUpperIndexMappedToValidLowerIndex(const UpIdx& /* idx_up */)
    {
        return true;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return is_known_at_compile_time<UpLengths>::value;
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("Modulus, ");
        printf("up_lengths_");
        print_multi_index(up_lengths_);
        printf("}");
    }
};
} // namespace ck
