#include <metal_stdlib>
using namespace metal;

kernel void square(device float *data [[buffer(0)]],
                   uint idx [[thread_position_in_grid]]) {
    data[idx] = data[idx] * data[idx];
}

kernel void inc_inplace(device float *data [[buffer(0)]],
                        uint idx [[thread_position_in_grid]]) {
    data[idx] = data[idx] + 1.0;
}
