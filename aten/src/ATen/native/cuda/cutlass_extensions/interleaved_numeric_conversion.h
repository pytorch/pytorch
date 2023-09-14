/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*!
    \file
    \brief Boost-like numeric conversion operator for int8 and CUTLASS int4b_t interleaved in a register
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/numeric_types.h"

namespace cutlass {

// This converter is meant to be used with data interleaved in a 32-bit register where the even elements are in the low
// bits and the odd elemeents are in the high bits of the register. In addition, it assumes elements were originally
// signed and had a bias of 2**(b-1) added (where b is the number of bits in the type) to make all numbers unsigned.
// This converter will uninterleave the data and subtract the bias while converting to the result type.
template<typename T, typename S, int N>
struct FastInterleavedAndBiasedNumericArrayConverter {
};

template<>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, 4> {
    using result_type = Array<half_t, 4>;
    using source_type = Array<uint8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

        static constexpr uint32_t mask_for_elt_01     = 0x5250;
        static constexpr uint32_t mask_for_elt_23     = 0x5351;
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[0]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
        asm volatile("prmt.b32 %0,%1,%2,%3;\n" : "=r"(h[1]) : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

        // Lastly, we subtract 1152 from our constructed number using fp16 math to get our signed integer as fp16.
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[1]) : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint8_t, N> {
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using result_type = Array<half_t, N>;
    using source_type = Array<uint8_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, 4> {
    using result_type = Array<bfloat16_t, 4>;
    using source_type = Array<uint8_t, 4>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

        uint32_t*      bf16_result_ptr = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8s             = reinterpret_cast<uint32_t const&>(source);

        static constexpr uint32_t fp32_base = 0x4B000000;
        float                     fp32_intermediates[4];

        // Construct FP32s, bfloat does not have enough mantissa for IADD trick
        uint32_t* fp32_intermediates_casted = reinterpret_cast<uint32_t*>(fp32_intermediates);
        fp32_intermediates_casted[0]        = __byte_perm(i8s, fp32_base, 0x7650);
        fp32_intermediates_casted[1]        = __byte_perm(i8s, fp32_base, 0x7652);
        fp32_intermediates_casted[2]        = __byte_perm(i8s, fp32_base, 0x7651);
        fp32_intermediates_casted[3]        = __byte_perm(i8s, fp32_base, 0x7653);

        // Subtract out fp32_base + 128 to make the unsigned integer signed.
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 4; ++ii) {
            fp32_intermediates[ii] -= 8388736.f;
        }

        // Truncate the fp32 representation and pack up as bfloat16s.
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < 2; ++ii) {
            bf16_result_ptr[ii] =
                __byte_perm(fp32_intermediates_casted[2 * ii + 0], fp32_intermediates_casted[2 * ii + 1], 0x7632);
        }
#else
        // Disable this on architectures older than Ampere since they lack hardware for bf16 mma. If one wishes to use
        // HMMA on older hardware, they should Convert directly to FP16 using FP16 converters.
        result.clear();  // Suppress compiler warning
        arch::device_breakpoint();
#endif
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint8_t, N> {
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using result_type = Array<bfloat16_t, N>;
    using source_type = Array<uint8_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, 8> {
    using result_type = Array<half_t, 8>;
    using source_type = Array<uint4b_t, 8>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;

        uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i4s = reinterpret_cast<uint32_t const&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        static constexpr uint32_t immLut                = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t BOTTOM_MASK           = 0x000f000f;
        static constexpr uint32_t TOP_MASK              = 0x00f000f0;
        static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

        // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
        // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
        // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
        // elt_67 to fp16 without having to shift them to the bottom bits before hand.

        // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
        // immediately before required.
        const uint32_t top_i4s = i4s >> 8;
        // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[1])
                     : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[2])
                     : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
        // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[3])
                     : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

        // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
        // half2 ctor. In this case, I chose performance reliability over code readability.

        // This is the half2 {1032, 1032} represented as an integer.
        static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
        // This is the half2 {1 / 16, 1 / 16} represented as an integer.
        static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
        // This is the half2 {-72, -72} represented as an integer.
        static constexpr uint32_t NEG_72 = 0xd480d480;

        // Finally, we construct the output numbers.
        // Convert elt_01
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_23
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
        // Convert elt_45
        asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
        // Convert elt_67
        asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<half_t, uint4b_t, N> {
    static constexpr int VEC_WIDTH = 8;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

    using result_type = Array<half_t, N>;
    using source_type = Array<uint4b_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, 8> {
    using result_type = Array<bfloat16_t, 8>;
    using source_type = Array<uint4b_t, 8>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        result_type result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))

        uint32_t*      h          = reinterpret_cast<uint32_t*>(&result);
        uint32_t const source_i4s = reinterpret_cast<uint32_t const&>(source);

        // First, we extract the i4s and construct an intermediate fp16 number.
        static constexpr uint32_t immLut                 = (0xf0 & 0xcc) | 0xaa;
        static constexpr uint32_t MASK                   = 0x000f000f;
        static constexpr uint32_t I4s_TO_BF16s_MAGIC_NUM = 0x43004300;

        // We don't have enough mantissa to remove as much shift overhead as FP16, so we must loop.
        // No shift needed for first item.
        uint32_t i4s = source_i4s;
        asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                     : "=r"(h[0])
                     : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 1; ii < result_type::kElements / 2; ++ii) {
            i4s >>= sizeof_bits<typename source_type::Element>::value;
            // (i4s & 0x000f000f) | 0x43004300
            asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                         : "=r"(h[ii])
                         : "r"(i4s), "n"(MASK), "n"(I4s_TO_BF16s_MAGIC_NUM), "n"(immLut));
        }

        // This is the BF16 {-136, -136} represented as an integer.
        static constexpr uint32_t BF16_BIAS = 0xC308C308;
        static constexpr uint32_t BF16_ONE  = 0x3F803F80;

        // Finally, we construct the output numbers.
        CUTLASS_PRAGMA_UNROLL
        for (int ii = 0; ii < result_type::kElements / 2; ++ii) {
            // Since this section is for Ampere+, we use bf16 fma to do the bias subtraction
            asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(h[ii]) : "r"(h[ii]), "r"(BF16_ONE), "r"(BF16_BIAS));
        }
#else
        // Disable this on architectures older than Ampere since they lack hardware for bf16 mma. If one wishes to use
        // HMMA on older hardware, they should Convert directly to FP16 using FP16 converters.
        arch::device_breakpoint();
        result.clear();  // Suppress compiler warning.
#endif
        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

template<int N>
struct FastInterleavedAndBiasedNumericArrayConverter<bfloat16_t, uint4b_t, N> {
    static constexpr int VEC_WIDTH = 8;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 8.");

    using result_type = Array<bfloat16_t, N>;
    using source_type = Array<uint4b_t, N>;

    CUTLASS_DEVICE
    static result_type convert(source_type const& source)
    {
        using scalar_result_type = typename result_type::Element;
        using scalar_source_type = typename source_type::Element;
        FastInterleavedAndBiasedNumericArrayConverter<scalar_result_type, scalar_source_type, VEC_WIDTH>
            convert_vector_;

        result_type result;
        using vec_result = Array<scalar_result_type, VEC_WIDTH>;
        using vec_source = Array<scalar_source_type, VEC_WIDTH>;

        vec_result*       result_ptr = reinterpret_cast<vec_result*>(&result);
        vec_source const* source_ptr = reinterpret_cast<vec_source const*>(&source);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N / VEC_WIDTH; ++i) {
            result_ptr[i] = convert_vector_(source_ptr[i]);
        }

        return result;
    }

    CUTLASS_DEVICE
    result_type operator()(source_type const& s)
    {
        return convert(s);
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////