//TEST_IGNORE_FILE:

#version 450
layout(row_major) uniform;
layout(row_major) buffer;

layout(rgba32f)
layout(binding = 0)
uniform image2D t_0;

void writeColor_0(vec3 v_0)
{
    const uvec2 _S1 = uvec2(0U, 0U);

    vec4 _S2 = (imageLoad((t_0), ivec2((_S1))));

    vec3 _S3 = _S2.xyz + v_0;

    ivec2 _S4 = ivec2(_S1);

    vec4 _S5 = imageLoad(t_0,_S4);

    vec4 _S6 = _S5;
    _S6.xyz = _S3;

    imageStore(t_0,_S4,_S6);
    return;
}


layout(location = 0)
out vec4 _S7;

void main()
{
    writeColor_0(vec3(1.00000000000000000000));
    _S7 = vec4(0);
    return;
}