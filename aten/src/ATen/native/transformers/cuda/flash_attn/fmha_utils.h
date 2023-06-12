/******************************************************************************
 * Copyright (c) 2011-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

#define FMHA_CHECK_CUDA( call )                                                                    \
    do {                                                                                           \
        cudaError_t status_ = call;                                                                \
        if( status_ != cudaSuccess ) {                                                             \
            fprintf( stderr,                                                                       \
                     "CUDA error (%s:%d): %s\n",                                                   \
                     __FILE__,                                                                     \
                     __LINE__,                                                                     \
                     cudaGetErrorString( status_ ) );                                              \
            exit( 1 );                                                                             \
        }                                                                                          \
    } while( 0 )

////////////////////////////////////////////////////////////////////////////////////////////////////

enum Data_type { DATA_TYPE_FP16, DATA_TYPE_BF16, DATA_TYPE_FP32, DATA_TYPE_INT32, DATA_TYPE_INT8 };

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_alpha( uint32_t &alpha, float norm, Data_type dtype ) {
    if( dtype == DATA_TYPE_FP16 ) {
        half x = __float2half_rn( norm );
        uint16_t h = reinterpret_cast<const uint16_t &>( x );
        ushort2 h2 = { h, h };
        alpha = reinterpret_cast<const uint32_t &>( h2 );
    } else if( dtype == DATA_TYPE_BF16 ) {
        __nv_bfloat16 x = __float2bfloat16( norm );
        uint16_t h = reinterpret_cast<const uint16_t &>( x );
        ushort2 h2 = { h, h };
        alpha = reinterpret_cast<const uint32_t &>( h2 );
    } else if( dtype == DATA_TYPE_FP32 ) {
        alpha = reinterpret_cast<const uint32_t &>( norm );
    } else if( dtype == DATA_TYPE_INT32 ) {
        int32_t inorm = static_cast<int32_t>( norm );
        alpha = reinterpret_cast<const uint32_t &>( inorm );
    } else {
        assert( false );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline size_t get_size_in_bytes( size_t n, Data_type dtype ) {
    switch( dtype ) {
    case DATA_TYPE_FP32:
        return n * 4;
    case DATA_TYPE_FP16:
        return n * 2;
    case DATA_TYPE_BF16:
        return n * 2;
    case DATA_TYPE_INT32:
        return n * 4;
    case DATA_TYPE_INT8:
        return n;
    default:
        assert( false );
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
