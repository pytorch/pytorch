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
    metal::array<constant void *, 16>  indexArray [[ id(0) ]];
};

template<typename T>
kernel void index_select(
    constant IndexAB  & indexAB           [[buffer(0)]],
    constant void     * indexSizes        [[buffer(1)]],
    constant void     * indexStrides      [[buffer(2)]],
    constant uint3    * offsets           [[buffer(3)]],
    constant void     * inputData         [[buffer(4)]],
    device   void     * outputData        [[buffer(5)]],
    uint thread_index [[thread_position_in_grid]]) {
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x);
    constant T * in  = (constant T*)((constant char*)inputData  + offsets[thread_index].y + offset);
    *out = *in;
}

template<typename T>
kernel void index_put(
    constant IndexAB  & indexAB           [[buffer(0)]],
    constant void     * indexSizes        [[buffer(1)]],
    constant void     * indexStrides      [[buffer(2)]],
    constant uint3    * offsets           [[buffer(3)]],
    constant void     * inputData         [[buffer(4)]],
    device   void     * outputData        [[buffer(5)]],
    uint thread_index [[thread_position_in_grid]]) {

    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
     }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    constant T * in  = (constant T*)((constant char*)inputData  + offsets[thread_index].y);
    *out = *in;
}

#define REGISTER_INDEX_OP(DTYPE_SIZE, DTYPE, INDEX_OP_TYPE)     \
template                                                        \
[[host_name("index_" #INDEX_OP_TYPE "_" #DTYPE_SIZE)]]          \
kernel void index_ ## INDEX_OP_TYPE<DTYPE>(                     \
    constant IndexAB & indexAB           [[buffer(0)]],         \
    constant void    * indexSizes        [[buffer(1)]],         \
    constant void    * indexStrides      [[buffer(2)]],         \
    constant uint3   * offsets           [[buffer(3)]],         \
    constant void    * inputData         [[buffer(4)]],         \
    device   void    * outputData        [[buffer(5)]],         \
    uint thread_index [[thread_position_in_grid]]);

#define REGISTER_INDEX_OP_ALL_DTYPES(INDEX_OP_TYPE)     \
    REGISTER_INDEX_OP(8bit,  char,  INDEX_OP_TYPE);     \
    REGISTER_INDEX_OP(16bit, short, INDEX_OP_TYPE);     \
    REGISTER_INDEX_OP(32bit, int,   INDEX_OP_TYPE);     \
    REGISTER_INDEX_OP(64bit, long,  INDEX_OP_TYPE);

REGISTER_INDEX_OP_ALL_DTYPES(select);
REGISTER_INDEX_OP_ALL_DTYPES(put);

kernel void kernel_index_offsets(constant packed_uint3 * strides         [[buffer(0)]],
                                 device uint3          * data_offsets    [[buffer(1)]],
                                 constant uint         * iter_shape      [[buffer(2)]],
                                 constant uint         & num_dimensions  [[buffer(3)]],
                                 constant uint         & num_offsets     [[buffer(4)]],
                                 uint thread_index [[thread_position_in_grid]]) {
    uint32_t idx = thread_index;
    for (uint32_t dim = 0; dim < num_dimensions; dim++) {
        uint32_t remainder = idx % iter_shape[dim];
        idx /= iter_shape[dim];

        for (uint32_t offset = 0; offset < num_offsets; offset++)
            data_offsets[thread_index][offset] += remainder * strides[dim][offset];
    }
}

template<typename T, typename E>
kernel void index_put_accumulate_native_dtypes(constant IndexAB & indexAB      [[buffer(0)]],
                                               constant void    * indexSizes   [[buffer(1)]],
                                               constant void    * indexStrides [[buffer(2)]],
                                               constant uint3   * offsets      [[buffer(3)]],
                                               constant void    * inputData    [[buffer(4)]],
                                               device       void    * outputData   [[buffer(5)]],
                                               uint thread_index [[thread_position_in_grid]]) {
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
    }
    device T * out = (device T*)((device char*)outputData + offsets[thread_index].x + offset);
    constant E * in  = (constant E*)((constant char*)inputData  + offsets[thread_index].y);
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
kernel void atomic_index_put_accumulate(constant IndexAB & indexAB           [[buffer(0)]],
                                        constant void    * indexSizes        [[buffer(1)]],
                                        constant void    * indexStrides      [[buffer(2)]],
                                        constant uint3   * offsets           [[buffer(3)]],
                                        constant void    * inputData         [[buffer(4)]],
                                        device   void    * outputData        [[buffer(5)]],
                                        uint thread_index [[thread_position_in_grid]]) {
    constant int64_t * index_sizes   = (constant int64_t *)indexSizes;
    constant int64_t * index_strides = (constant int64_t *)indexStrides;
    int64_t offset = 0;
    for (uint32_t i = 0; i < num_indices; i++) {
        int64_t index = ((constant int64_t*)(indexAB.indexArray[i]))[offsets[thread_index].z / sizeof(int64_t)];
        if (index < 0) {
            index += index_sizes[i];
        }
        offset += index * index_strides[i];
    }
    device void * out = (device void*)((device char*)outputData + offsets[thread_index].x + offset);
    constant T  * in  = (constant T*)((constant char*)inputData + offsets[thread_index].y);
    atomic_fetch_add_relaxed<T>(out, *in);
}

template
[[host_name("index_put_accumulate_32bit_float")]]
kernel void atomic_index_put_accumulate<float>(constant IndexAB & indexAB      [[buffer(0)]],
                                               constant void    * indexSizes   [[buffer(1)]],
                                               constant void    * indexStrides [[buffer(2)]],
                                               constant uint3   * offsets      [[buffer(3)]],
                                               constant void    * inputData    [[buffer(4)]],
                                               device   void    * outputData   [[buffer(5)]],
                                               uint thread_index [[thread_position_in_grid]]);
template
[[host_name("index_put_accumulate_32bit_int")]]
kernel void index_put_accumulate_native_dtypes<atomic_int, int>(constant IndexAB & indexAB      [[buffer(0)]],
                                                                constant void    * indexSizes   [[buffer(1)]],
                                                                constant void    * indexStrides [[buffer(2)]],
                                                                constant uint3   * offsets      [[buffer(3)]],
                                                                constant void    * inputData    [[buffer(4)]],
                                                                device   void    * outputData   [[buffer(5)]],
                                                                uint thread_index [[thread_position_in_grid]]);
)INDEX_METAL";
}
}
