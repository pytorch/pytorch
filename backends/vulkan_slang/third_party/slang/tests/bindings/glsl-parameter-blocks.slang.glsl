#version 450
layout(row_major) uniform;
layout(row_major) buffer;
struct Test_0
{
    vec4 a_0;
};

layout(binding = 0)
layout(std140) uniform _S1
{
    vec4 a_0;
}gTest_0;
layout(binding = 1)
uniform texture2D gTest_t_0;

layout(binding = 2)
uniform sampler gTest_s_0;

layout(location = 0)
out vec4 main_0;

layout(location = 0)
in vec2 uv_0;

void main()
{
    main_0 = gTest_0.a_0 + (texture(sampler2D(gTest_t_0,gTest_s_0), (uv_0)));
    return;
}

