//TEST_IGNORE_FILE:
#version 450
layout(row_major) uniform;
layout(row_major) buffer;
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(r16f)
layout(binding = 1)
uniform image2D halfTexture_0;

layout(rg16f)
layout(binding = 2)
uniform image2D halfTexture2_0;

layout(rgba16f)
layout(binding = 3)
uniform image2D halfTexture4_0;

layout(std430, binding = 0) buffer StructuredBuffer_int_t_0 {
    int _data[];
} outputBuffer_0;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
void main()
{
    ivec2 pos_0 = ivec2(gl_GlobalInvocationID.xy);

    int _S1 = pos_0.y;

    int _S2 = pos_0.x;

    ivec2 _S3 = ivec2(uvec2(ivec2(3 - _S1, 3 - _S2)));

    float16_t _S4 = (float16_t(imageLoad((halfTexture_0), (_S3)).x));
    f16vec2 _S5 = (f16vec2(imageLoad((halfTexture2_0), (_S3)).xy));
    f16vec4 _S6 = (f16vec4(imageLoad((halfTexture4_0), (_S3))));

    ivec2 _S7 = ivec2(uvec2(pos_0));

    imageStore((halfTexture_0), (_S7), f16vec4(_S5.x + _S5.y, float16_t(0), float16_t(0), float16_t(0)));
    imageStore((halfTexture2_0), (_S7), f16vec4(_S6.xy, float16_t(0), float16_t(0)));
    imageStore((halfTexture4_0), (_S7), f16vec4(_S5, _S4, _S4));

    int index_0 = _S2 + _S1 * 4;
    outputBuffer_0._data[uint(index_0)] = index_0;
    return;
}
