#version 450
layout(row_major) uniform;
layout(row_major) buffer;
layout(std430, binding = 0) buffer StructuredBuffer_int_t_0 {
    int _data[];
} a_0;
layout(std430, binding = 1) buffer StructuredBuffer_int_t_1 {
    int _data[];
} b_0[3];
layout(std430, binding = 2) buffer StructuredBuffer_int_t_2 {
    int _data[];
} c_0[4][3];
int f_0(uint _S1)
{
    return a_0._data[_S1];
}

int f_1(uint _S2, uint _S3)
{
    return b_0[_S2]._data[_S3];
}

int g_0(uint _S4, uint _S5)
{
    return b_0[_S4]._data[_S5];
}

int g_1(uint _S6, uint _S7, uint _S8)
{
    return c_0[_S6][_S7]._data[_S8];
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint ii_0 = gl_GlobalInvocationID.x;
    uint jj_0 = gl_GlobalInvocationID.y;
    a_0._data[ii_0] = f_0(ii_0) + f_0(jj_0) + f_1(ii_0, jj_0) + g_0(ii_0, jj_0) + g_1(ii_0, jj_0, gl_GlobalInvocationID.z);
    return;
}

