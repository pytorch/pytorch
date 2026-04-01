//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage fragment -entry main -allow-glsl -emit-spirv-directly
#version 450 core
layout(location = 0) out mediump vec4 o_color;
layout(location = 0) in highp vec2 v_texCoord;
layout(location = 1) in highp float v_lodBias;
layout(set = 0, binding = 0) uniform highp usampler2D u_sampler;
layout(set = 0, binding = 1) uniform buf0 { highp vec4 u_scale; };
layout(set = 0, binding = 2) uniform buf1 { highp vec4 u_bias; };

void main()
{
    o_color = vec4(texelFetch(u_sampler, ivec2(v_texCoord), int(v_lodBias))) * u_scale + u_bias;
    // CHECK: OpEntryPoint
}

