//TEST_IGNORE_FILE:
#version 450
layout(row_major) uniform;
layout(row_major) buffer;

layout(location = 0)
out vec4 _S1;

layout(location = 0)
in vec4 _S2;

flat layout(location = 1)
in uint _S3;


void main()
{
    vec4 result_0;
    if((_S3 & 1U) != 0U)
    {
        result_0 = _S2 + 1.00000000000000000000;
    }
    else
    {
        result_0 = _S2;
    }
    _S1 = result_0;
    return;
}
