#version 450
layout(row_major) uniform;
layout(row_major) buffer;
struct SLANG_ParameterGroup_C_0
{
    uint index_0;
};

layout(binding = 0)
layout(std140) uniform _S1
{
    uint index_0;
}C_0;
struct S_0
{
    vec4 f_0;
};
layout(binding = 1)
layout(std140) uniform _S2
{
    vec4 f_0;
}cb_0[3];
layout(std430, binding = 2) readonly buffer StructuredBuffer_S_t_0 {
    S_0 _data[];
} sb1_0[4];
layout(std430, binding = 3) buffer StructuredBuffer_float4_t_0 {
    vec4 _data[];
} sb2_0[5];
layout(std430, binding = 4) readonly buffer StructuredBuffer_uint_t_0 {
    uint _data[];
} bb_0[6];
layout(location = 0)
out vec4 main_0;

void main()
{
    main_0 = cb_0[C_0.index_0].f_0 + sb1_0[C_0.index_0]._data[C_0.index_0].f_0 + sb2_0[C_0.index_0]._data[C_0.index_0] + vec4(float(bb_0[C_0.index_0]._data[int(C_0.index_0 * 4U) / 4]));
    return;
}
