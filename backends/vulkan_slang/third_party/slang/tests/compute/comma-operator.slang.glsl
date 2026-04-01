// comma-operator.slang.glsl
#version 450

//TEST_IGNORE_FILE:

layout(std430, binding = 0) buffer StructuredBuffer_int_t_0 {
    int _data[];
} outputBuffer_0;

int test_0(int inVal_0)
{
    return inVal_0 * 2 + 1;
}

layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint tid_0 = gl_GlobalInvocationID.x;
    outputBuffer_0._data[tid_0] = test_0(outputBuffer_0._data[tid_0]);
    return;
}
