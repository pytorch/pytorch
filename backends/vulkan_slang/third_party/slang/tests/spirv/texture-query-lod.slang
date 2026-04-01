//TEST:SIMPLE(filecheck=CHK): -target spirv -entry main -stage fragment
//TEST:SIMPLE(filecheck=CHK): -target spirv-asm -entry main -stage fragment

#version 450 core

layout (location = 0) in highp vec2 uv;
layout (location = 0) out mediump vec4 fragment;
layout (set = 0, binding = 0) uniform highp isampler2D sampler;

void main()
{
    // CHK: OpCapability ImageQuery
    fragment = vec4(textureQueryLod(sampler, uv), 0.0, 1.0);
}
