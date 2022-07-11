#pragma once

namespace at {
namespace mps {

static const char * indexing_metal_shaders = R"INDEX_METAL(
#include <metal_stdlib>
using namespace metal;

constant int64_t  storage_offset  [[function_constant(0)]];
constant uint32_t num_indices     [[function_constant(1)]];

struct IndexAB {
    // Allow up to 30 indices
    metal::array<device void *, 30>  indexArray [[ id(0) ]];
};

template<typename T>
kernel void index_select(device const IndexAB & indexAB           [[buffer(0)]],
                         device const void    * indexSizes        [[buffer(1)]],
                         device const void    * indexStrides      [[buffer(2)]],
                         device const uint3   * offsets           [[buffer(3)]],
                         device const void    * inputData         [[buffer(4)]],
                         device void          * outputData        [[buffer(5)]],
                         uint thread_index [[thread_position_in_grid]]) {

    device const int64_t * index_sizes   = (device const int64_t *)indexSizes;
    device const int64_t * index_strides = (device const int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((device const int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x);
    device T * in  = (device T*)((device char*)inputData  + offsets[thread_index].y + offset + storage_offset * sizeof(T));
    *out = *in;
}

template
[[host_name("index_select_float")]]
kernel void index_select<float>(device const IndexAB & indexAB       [[buffer(0)]],
                                device const void    * indexSizes    [[buffer(1)]],
                                device const void    * indexStrides  [[buffer(2)]],
                                device const uint3   * offsets       [[buffer(3)]],
                                device const void    * inputData     [[buffer(4)]],
                                device void          * outputData    [[buffer(5)]],
                                uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_half")]]
kernel void index_select<half>(device const IndexAB & indexAB       [[buffer(0)]],
                                device const void   * indexSizes    [[buffer(1)]],
                                device const void   * indexStrides  [[buffer(2)]],
                                device const uint3  * offsets       [[buffer(3)]],
                                device const void   * inputData     [[buffer(4)]],
                                device void         * outputData    [[buffer(5)]],
                                uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_int32")]]
kernel void index_select<int32_t>(device const IndexAB & indexAB       [[buffer(0)]],
                                  device const void    * indexSizes    [[buffer(1)]],
                                  device const void    * indexStrides  [[buffer(2)]],
                                  device const uint3   * offsets       [[buffer(3)]],
                                  device const void    * inputData     [[buffer(4)]],
                                  device void          * outputData    [[buffer(5)]],
                                  uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_int64")]]
kernel void index_select<int64_t>(device const IndexAB & indexAB       [[buffer(0)]],
                                  device const void    * indexSizes    [[buffer(1)]],
                                  device const void    * indexStrides  [[buffer(2)]],
                                  device const uint3   * offsets       [[buffer(3)]],
                                  device const void    * inputData     [[buffer(4)]],
                                  device void          * outputData    [[buffer(5)]],
                                  uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_int16")]]
kernel void index_select<int16_t>(device const IndexAB & indexAB       [[buffer(0)]],
                                  device const void    * indexSizes    [[buffer(1)]],
                                  device const void    * indexStrides  [[buffer(2)]],
                                  device const uint3   * offsets       [[buffer(3)]],
                                  device const void    * inputData     [[buffer(4)]],
                                  device void          * outputData    [[buffer(5)]],
                                  uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_uint8")]]
kernel void index_select<uint8_t>(device const IndexAB & indexAB       [[buffer(0)]],
                                  device const void    * indexSizes    [[buffer(1)]],
                                  device const void    * indexStrides  [[buffer(2)]],
                                  device const uint3   * offsets       [[buffer(3)]],
                                  device const void    * inputData     [[buffer(4)]],
                                  device void          * outputData    [[buffer(5)]],
                                  uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_select_bool")]]
kernel void index_select<bool>(device const IndexAB & indexAB       [[buffer(0)]],
                               device const void    * indexSizes    [[buffer(1)]],
                               device const void    * indexStrides  [[buffer(2)]],
                               device const uint3   * offsets       [[buffer(3)]],
                               device const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
)INDEX_METAL";

}
}