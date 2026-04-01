#version 450

layout(r32ui)
layout(binding = 0)
uniform uimage2D t_0;

layout(location = 0)
out vec4 main_0;

void main()
{
    uint u_0;
    u_0 = imageAtomicAdd(t_0, ivec2(uvec2(0)), 1);
    main_0 = vec4(u_0);
    return;
}
