#pragma once

namespace at {
namespace mps {

static const char * indexing_metal_shaders = R"INDEX_METAL(
#include <metal_stdlib>
#include <metal_atomic>

using namespace metal;

constant uint32_t num_indices            [[function_constant(0)]];

struct IndexAB {
    // Allow up to 16 indices
    metal::array<device void *, 16>  indexArray [[ id(0) ]];
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
    device T * in  = (device T*)((device char*)inputData  + offsets[thread_index].y + offset);
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

template<typename T>
kernel void index_put(device const IndexAB & indexAB           [[buffer(0)]],
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
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    device T * in  = (device T*)((device char*)inputData  + offsets[thread_index].y);
    *out = *in;
}

template
[[host_name("index_put_float")]]
kernel void index_put<float>(device const IndexAB & indexAB       [[buffer(0)]],
                             device const void    * indexSizes    [[buffer(1)]],
                             device const void    * indexStrides  [[buffer(2)]],
                             device const uint3   * offsets       [[buffer(3)]],
                             device const void    * inputData     [[buffer(4)]],
                             device void          * outputData    [[buffer(5)]],
                             uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_half")]]
kernel void index_put<half>(device const IndexAB & indexAB       [[buffer(0)]],
                            device const void    * indexSizes    [[buffer(1)]],
                            device const void    * indexStrides  [[buffer(2)]],
                            device const uint3   * offsets       [[buffer(3)]],
                            device const void    * inputData     [[buffer(4)]],
                            device void          * outputData    [[buffer(5)]],
                            uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_int32")]]
kernel void index_put<int32_t>(device const IndexAB & indexAB       [[buffer(0)]],
                               device const void    * indexSizes    [[buffer(1)]],
                               device const void    * indexStrides  [[buffer(2)]],
                               device const uint3   * offsets       [[buffer(3)]],
                               device const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_int64")]]
kernel void index_put<int64_t>(device const IndexAB & indexAB       [[buffer(0)]],
                               device const void    * indexSizes    [[buffer(1)]],
                               device const void    * indexStrides  [[buffer(2)]],
                               device const uint3   * offsets       [[buffer(3)]],
                               device const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_int16")]]
kernel void index_put<int16_t>(device const IndexAB & indexAB       [[buffer(0)]],
                               device const void    * indexSizes    [[buffer(1)]],
                               device const void    * indexStrides  [[buffer(2)]],
                               device const uint3   * offsets       [[buffer(3)]],
                               device const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_uint8")]]
kernel void index_put<uint8_t>(device const IndexAB & indexAB       [[buffer(0)]],
                               device const void    * indexSizes    [[buffer(1)]],
                               device const void    * indexStrides  [[buffer(2)]],
                               device const uint3   * offsets       [[buffer(3)]],
                               device const void    * inputData     [[buffer(4)]],
                               device void          * outputData    [[buffer(5)]],
                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_bool")]]
kernel void index_put<bool>(device const IndexAB & indexAB       [[buffer(0)]],
                            device const void    * indexSizes    [[buffer(1)]],
                            device const void    * indexStrides  [[buffer(2)]],
                            device const uint3   * offsets       [[buffer(3)]],
                            device const void    * inputData     [[buffer(4)]],
                            device void          * outputData    [[buffer(5)]],
                            uint thread_index [[thread_position_in_grid]]);

template<typename T, typename E>
kernel void index_put_accumulate_native_dtypes(device const IndexAB & indexAB      [[buffer(0)]],
                                               device const void    * indexSizes   [[buffer(1)]],
                                               device const void    * indexStrides [[buffer(2)]],
                                               device const uint3   * offsets      [[buffer(3)]],
                                               device const void    * inputData    [[buffer(4)]],
                                               device       void    * outputData   [[buffer(5)]],
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
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    device E * in  = (device E*)((device char*)inputData  + offsets[thread_index].y);
    atomic_fetch_add_explicit(out, *in, memory_order_relaxed);
}

template<typename T>
__attribute__((__always_inline__)) void atomic_fetch_add_relaxed(device void * addr, T value) {
    device atomic_uint* uintAddr = (device atomic_uint*)addr;
    uint expected = atomic_load_explicit(uintAddr, memory_order_relaxed);
    T updated = as_type<T>(expected) + value;
    while (!atomic_compare_exchange_weak_explicit(uintAddr, &expected, as_type<uint>(updated), memory_order_relaxed, memory_order_relaxed)) {
        updated = as_type<T>(expected) + value;
    }
}

template<typename T>
kernel void atomic_index_put_accumulate(device const IndexAB & indexAB           [[buffer(0)]],
                                        device const void    * indexSizes        [[buffer(1)]],
                                        device const void    * indexStrides      [[buffer(2)]],
                                        device const uint3   * offsets           [[buffer(3)]],
                                        device const void    * inputData         [[buffer(4)]],
                                        device       void    * outputData        [[buffer(5)]],
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
    device void * out = (device void*)((device char*)outputData + offsets[thread_index].x + offset);
    device T * in  = (device T*)((device char*)inputData  + offsets[thread_index].y);
    atomic_fetch_add_relaxed<T>(out, *in);
}

template
[[host_name("index_put_accumulate_float")]]
kernel void atomic_index_put_accumulate<float>(device const IndexAB & indexAB             [[buffer(0)]],
                                               device const void    * indexSizes   [[buffer(1)]],
                                               device const void    * indexStrides [[buffer(2)]],
                                               device const uint3   * offsets      [[buffer(3)]],
                                               device const void    * inputData    [[buffer(4)]],
                                               device void          * outputData   [[buffer(5)]],
                                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_accumulate_int32")]]
kernel void index_put_accumulate_native_dtypes<atomic_int, int32_t>(device const IndexAB & indexAB      [[buffer(0)]],
                                                                    device const void    * indexSizes   [[buffer(1)]],
                                                                    device const void    * indexStrides [[buffer(2)]],
                                                                    device const uint3   * offsets      [[buffer(3)]],
                                                                    device const void    * inputData    [[buffer(4)]],
                                                                    device void          * outputData   [[buffer(5)]],
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
