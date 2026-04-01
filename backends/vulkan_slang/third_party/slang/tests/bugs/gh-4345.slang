//TEST:SIMPLE(filecheck=CHECK):-target spirv-asm -entry main -stage vertex -allow-glsl

// CHECK: OpDecorate %{{.*}} Flat

#version 450
layout(set = 1, binding = 0) uniform highp isampler1D texSampler[8];

layout(location = 0) in highp vec4 a_position;
layout(location = 1) in highp float a_coords;
layout(location = 0) flat out highp ivec4 vtx_out_result0;
layout(location = 1) flat out highp ivec4 vtx_out_result1;
layout(location = 2) flat out highp ivec4 vtx_out_result2;
layout(location = 3) flat out highp ivec4 vtx_out_result3;

void main (void)
{
        gl_Position = a_position;
        gl_PointSize = 1.0;
        highp float coords = a_coords;
        highp ivec4 result0;
        highp ivec4 result1;
        highp ivec4 result2;
        highp ivec4 result3;
        result0 = textureLod(texSampler[3], coords, 0.0);
        result1 = textureLod(texSampler[2], coords, 0.0);
        result2 = textureLod(texSampler[7], coords, 0.0);
        result3 = textureLod(texSampler[2], coords, 0.0);
        vtx_out_result0 = result0;
        vtx_out_result1 = result1;
        vtx_out_result2 = result2;
        vtx_out_result3 = result3;
}