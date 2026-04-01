#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(row_major) uniform;
layout(row_major) buffer;
layout(binding = 0)
uniform texture2DMS tex_0;

layout(std430, binding = 1) buffer StructuredBuffer_float4_t_0 {
    vec4 _data[];
} outBuffer_0;
layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
void main()
{
    vec4 _S2 = (texelFetch((tex_0), (ivec2(gl_WorkGroupID.xy)), (0)));
    ((outBuffer_0)._data[(0U)]) = _S2;
    return;
}
