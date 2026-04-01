#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(row_major) uniform;
layout(row_major) buffer;
struct SLANG_ParameterGroup_C_0
{
    vec2 uv_0;
    uint index_0;
};

layout(binding = 2)
layout(std140) uniform _S1
{
    vec2 uv_0;
    uint index_0;
}C_0;
layout(binding = 0)
uniform texture2D  t_0[];

layout(binding = 1)
uniform sampler s_0;

layout(location = 0)
out vec4 main_0;

void main()
{
    main_0 = (texture(sampler2D(t_0[C_0.index_0],s_0), (C_0.uv_0)));
    return;
}

