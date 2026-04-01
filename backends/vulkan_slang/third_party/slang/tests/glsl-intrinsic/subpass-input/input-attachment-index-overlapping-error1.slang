//TEST:SIMPLE(filecheck=CHECK): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl
#version 450
// CHECK: warning 39001
buffer MyBlockName
{
    uvec4 idata[2];
} keepAliveBuffer;

layout(location = 0) in highp vec4 a_position;
layout(location = 0) in highp vec4 b_position;

layout (input_attachment_index = 2, set = 0, binding = 0) uniform isubpassInput isubpass;
layout (input_attachment_index = 2, set = 0, binding = 1) uniform isubpassInputMS isubpassMS;

layout (location = 0) out vec4 outColor;

void main() {
    keepAliveBuffer.idata[0] = subpassLoad(isubpass);
    keepAliveBuffer.idata[1] = subpassLoad(isubpassMS, 0);

    outColor = vec4(0);
}