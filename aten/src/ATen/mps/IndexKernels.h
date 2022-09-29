#pragma once

namespace at {
namespace mps {

static const char * indexing_metal_shaders = R"INDEX_METAL(
#include <metal_stdlib>
using namespace metal;

constant uint32_t num_indices            [[function_constant(0)]];

struct IndexAB {
    // Allow up to 16 indices
    metal::array<constant const void *, 16>  indexArray [[ id(0) ]];
};

template<typename T>
kernel void index_select(
    constant const IndexAB  & indexAB           [[buffer(0)]],
    constant const void     * indexSizes        [[buffer(1)]],
    constant const void     * indexStrides      [[buffer(2)]],
    constant const uint3    * offsets           [[buffer(3)]],
    constant const void     * inputData         [[buffer(4)]],
    device   void           * outputData        [[buffer(5)]],
    uint thread_index [[thread_position_in_grid]]) {

    constant const int64_t * index_sizes   = (constant const int64_t *)indexSizes;
    constant const int64_t * index_strides = (constant const int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant const int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x);
    constant const T * in  = (constant const T*)((constant const char*)inputData  + offsets[thread_index].y + offset);
    *out = *in;
}

template
[[host_name("index_select_float")]]
kernel void index_select<float>(constant const IndexAB & indexAB       [[buffer(0)]],
                                constant const void    * indexSizes    [[buffer(1)]],
                                constant const void    * indexStrides  [[buffer(2)]],
                                constant const uint3   * offsets       [[buffer(3)]],
                                constant const void    * inputData     [[buffer(4)]],
                                device void          * outputData    [[buffer(5)]],
                                uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_half")]]
kernel void index_select<half>(constant const IndexAB & indexAB       [[buffer(0)]],
                               constant const void   * indexSizes    [[buffer(1)]],
                               constant const void   * indexStrides  [[buffer(2)]],
                               constant const uint3  * offsets       [[buffer(3)]],
                               constant const void   * inputData     [[buffer(4)]],
                               device void         * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_long")]]
kernel void index_select<long>(constant const IndexAB & indexAB       [[buffer(0)]],
                               constant const void    * indexSizes    [[buffer(1)]],
                               constant const void    * indexStrides  [[buffer(2)]],
                               constant const uint3   * offsets       [[buffer(3)]],
                               constant const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_int")]]
kernel void index_select<int>(constant const IndexAB & indexAB       [[buffer(0)]],
                              constant const void    * indexSizes    [[buffer(1)]],
                              constant const void    * indexStrides  [[buffer(2)]],
                              constant const uint3   * offsets       [[buffer(3)]],
                              constant const void    * inputData     [[buffer(4)]],
                              device void          * outputData    [[buffer(5)]],
                              uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_short")]]
kernel void index_select<short>(constant const IndexAB & indexAB       [[buffer(0)]],
                                constant const void    * indexSizes    [[buffer(1)]],
                                constant const void    * indexStrides  [[buffer(2)]],
                                constant const uint3   * offsets       [[buffer(3)]],
                                constant const void    * inputData     [[buffer(4)]],
                                device void          * outputData    [[buffer(5)]],
                                uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_char")]]
kernel void index_select<char>(constant const IndexAB & indexAB       [[buffer(0)]],
                               constant const void    * indexSizes    [[buffer(1)]],
                               constant const void    * indexStrides  [[buffer(2)]],
                               constant const uint3   * offsets       [[buffer(3)]],
                               constant const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_uchar")]]
kernel void index_select<uchar>(constant const IndexAB & indexAB       [[buffer(0)]],
                                constant const void    * indexSizes    [[buffer(1)]],
                                constant const void    * indexStrides  [[buffer(2)]],
                                constant const uint3   * offsets       [[buffer(3)]],
                                constant const void    * inputData     [[buffer(4)]],
                                device void          * outputData    [[buffer(5)]],
                                uint thread_index [[thread_position_in_grid]]);

template
[[host_name("index_select_bool")]]
kernel void index_select<bool>(constant const IndexAB & indexAB       [[buffer(0)]],
                               constant const void    * indexSizes    [[buffer(1)]],
                               constant const void    * indexStrides  [[buffer(2)]],
                               constant const uint3   * offsets       [[buffer(3)]],
                               constant const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);

kernel void kernel_index_offsets(constant const packed_uint3 * strides         [[buffer(0)]],
                                 device uint3                * data_offsets    [[buffer(1)]],
                                 constant const uint         * iter_shape      [[buffer(2)]],
                                 constant const uint         & num_dimensions  [[buffer(3)]],
                                 constant const uint         & num_offsets     [[buffer(4)]],
                                 uint thread_index [[thread_position_in_grid]]) {
    uint32_t idx = thread_index;
    for (uint32_t dim = 0; dim < num_dimensions; dim++) {
        uint32_t remainder = idx % iter_shape[dim];
        idx /= iter_shape[dim];
        for (uint32_t offset = 0; offset < num_offsets; offset++)
            data_offsets[thread_index][offset] += remainder * strides[dim][offset];
    }
}
)INDEX_METAL";
}
}
