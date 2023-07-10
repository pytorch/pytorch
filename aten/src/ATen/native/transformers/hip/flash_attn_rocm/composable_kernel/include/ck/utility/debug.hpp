// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#ifndef UTILITY_DEBUG_HPP
#define UTILITY_DEBUG_HPP

namespace ck {
namespace debug {

namespace detail {
template <typename T, typename Enable = void>
struct PrintAsType;

template <typename T>
struct PrintAsType<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
{
    using type = float;
    __host__ __device__ static void Print(const T& p) { printf("%.3f ", static_cast<type>(p)); }
};

template <>
struct PrintAsType<ck::half_t, void>
{
    using type = float;
    __host__ __device__ static void Print(const ck::half_t& p)
    {
        printf("%.3f ", static_cast<type>(p));
    }
};

template <typename T>
struct PrintAsType<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
    using type = int;
    __host__ __device__ static void Print(const T& p) { printf("%d ", static_cast<type>(p)); }
};
} // namespace detail

// Print at runtime the data in shared memory in 128 bytes per row format given shared mem pointer
// and the number of elements. Can optionally specify strides between elements and how many bytes'
// worth of data per row.
//
// Usage example:
//
//   debug::print_shared(a_block_buf.p_data_, index_t(a_block_desc_k0_m_k1.GetElementSpaceSize()));
//
template <typename T, index_t element_stride = 1, index_t row_bytes = 128>
__device__ void print_shared(T const* p_shared, index_t num_elements)
{
    constexpr index_t row_elements = row_bytes / sizeof(T);
    static_assert((element_stride >= 1 && element_stride <= row_elements),
                  "element_stride should between [1, row_elements]");

    index_t wgid = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    index_t tid =
        (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    __syncthreads();

    if(tid == 0)
    {
        printf("\nWorkgroup id %d, bytes per row %d, element stride %d\n\n",
               wgid,
               row_bytes,
               element_stride);
        for(index_t i = 0; i < num_elements; i += row_elements)
        {
            printf("elem %5d: ", i);
            for(index_t j = 0; j < row_elements; j += element_stride)
            {
                detail::PrintAsType<T>::Print(p_shared[i + j]);
            }

            printf("\n");
        }
        printf("\n");
    }

    __syncthreads();
}

} // namespace debug
} // namespace ck

#endif // UTILITY_DEBUG_HPP
