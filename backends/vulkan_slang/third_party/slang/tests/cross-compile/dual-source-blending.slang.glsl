//dual-source-blending.slang.glsl
#version 450

layout(row_major) uniform;
layout(row_major) buffer;

layout(location = 0)
out vec4 main_a_0;

layout(location = 0, index = 1)
out vec4 main_b_0;

layout(location = 0)
in vec4 v_0;

struct FragmentOutput_0
{
    vec4 a_0;
    vec4 b_0;
};

void main()
{
    const vec4 _S1 = vec4(0.0, 0.0, 0.0, 0.0);
    FragmentOutput_0 f_0;

    f_0.a_0 = _S1;

    f_0.b_0 = _S1;
    f_0.a_0 = v_0;
    f_0.b_0 = v_0;
    FragmentOutput_0 _S2 = f_0;
    main_a_0 = f_0.a_0;
    main_b_0 = _S2.b_0;

    return;
}
