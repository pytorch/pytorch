// vk-push-constant.slang.glsl
#version 450

struct S_0
{
    vec4 v_0;
};

layout(push_constant)
layout(std140) uniform _S1
{
    vec4 v_0;
} x_0;

layout(binding = 0, set = 0)
layout(std140) uniform _S2
{
    vec4 v_0;
} y_0;

layout(location = 0)
out vec4 main_0;

void main()
{
    main_0 = x_0.v_0 + y_0.v_0;
    return;
}
