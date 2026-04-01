#version 450
#extension GL_EXT_shader_atomic_float : require
layout(row_major) uniform;
layout(row_major) buffer;

#line 11 "tests/slang-extension/atomic-float-byte-address-buffer-cross.slang"
layout(std430, binding = 1) buffer StructuredBuffer_float_t_0 {
    float _data[];
} anotherBuffer_0;

#line 11
layout(std430, binding = 0) buffer StructuredBuffer_float_t_1 {
    float _data[];
} outputBuffer_0;

#line 1264 "core.meta.slang"
void RWByteAddressBuffer_InterlockedAddF32_0(uint _S1, float _S2, out float _S3)
{

#line 391 "hlsl.meta.slang"
    float _S4 = (atomicAdd((outputBuffer_0._data[_S1 / 4U]), (_S2)));

#line 391
    _S3 = _S4;
    return;
}


#line 392
void RWByteAddressBuffer_InterlockedAddF32_1(uint _S5, float _S6)
{

#line 406
    float _S7 = (atomicAdd((outputBuffer_0._data[_S5 / 4U]), (_S6)));
    return;
}


#line 14 "tests/slang-extension/atomic-float-byte-address-buffer-cross.slang"
layout(local_size_x = 16, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 16
    uint tid_0 = gl_GlobalInvocationID.x;
    uint _S8 = tid_0 >> 2;

#line 17
    int idx_0 = int(tid_0 & 3U ^ _S8);

    float delta_0 = anotherBuffer_0._data[uint(idx_0 & 3)];

    float previousValue_0 = 0.0;

#line 21
    RWByteAddressBuffer_InterlockedAddF32_0(uint(idx_0 << 2), 1.0, previousValue_0);

#line 21
    RWByteAddressBuffer_InterlockedAddF32_1(uint(int(_S8) << 2), delta_0);

#line 27
    return;
}


