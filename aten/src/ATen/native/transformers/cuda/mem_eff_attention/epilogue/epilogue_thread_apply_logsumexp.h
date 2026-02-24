/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename Element, int ElementsPerAccess>
struct ArrayExponential {
  CUTLASS_HOST_DEVICE
  Array<Element, ElementsPerAccess> operator()(
      Array<Element, ElementsPerAccess> const& input) const {
    Array<Element, ElementsPerAccess> result;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ElementsPerAccess; ++i) {
      result[i] = expf(input[i]);
    }

    return result;
  }
};

template <int N>
struct dependent_false : std::false_type {};

template <int ElementsPerAccess>
struct ArrayExponential<half_t, ElementsPerAccess> {
  static_assert(
      dependent_false<ElementsPerAccess>::value,
      "ArrayExponential is not implemented for half_t");
};

template <int ElementsPerAccess>
struct ArrayExponential<bfloat16_t, ElementsPerAccess> {
  static_assert(
      dependent_false<ElementsPerAccess>::value,
      "ArrayExponential is not implemented for bfloat16_t");
};
} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies:
/// output <- (input - lse).exp()
template <
    typename ElementOutput_, // output
    typename ElementLSE_, // accumulator from LSE
    typename ElementAccumulator_, // accumulator from matmul
    typename ElementCompute_, // intermediate compute (and exp calculation)
    int ElementsPerAccess>
class ApplyLogSumExp {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementLSE = ElementLSE_;

  static int constexpr kElementsPerAccess = ElementsPerAccess;
  static int constexpr kCount = kElementsPerAccess;
  static constexpr ScaleType::Kind kScale =
      cutlass::epilogue::thread::ScaleType::NoBetaScaling;

  using FragmentOutput = Array<ElementOutput, kCount>;
  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = Array<ElementCompute, kElementsPerAccess>;
  using FragmentLSE = Array<ElementLSE, kElementsPerAccess>;
  using FragmentScaleBias = FragmentLSE; // Used by epilogue_smem_accumulator.h

 public:
  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ApplyLogSumExp() {}

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& AB,
      FragmentLSE const& scale_unused,
      // bias used as LSE
      FragmentLSE const& bias) const {
    FragmentCompute frag_AB = NumericArrayConverter<
        ElementCompute,
        ElementAccumulator,
        kElementsPerAccess>()(AB);
    FragmentCompute frag_lse_compute =
        NumericArrayConverter<ElementCompute, ElementLSE, kElementsPerAccess>()(
            bias);
    FragmentCompute frag_compute;

    minus<FragmentCompute> minus_lse;
    detail::ArrayExponential<ElementCompute, kElementsPerAccess> apply_exp;
    frag_compute = minus_lse(frag_AB, frag_lse_compute);
    frag_compute = apply_exp(frag_compute);

    return NumericArrayConverter<
        ElementOutput,
        ElementCompute,
        kElementsPerAccess>()(frag_compute);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
