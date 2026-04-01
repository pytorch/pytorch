//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl

#version 450

// CHECK_SPV-DAG: InputAttachmentIndex 0
// CHECK_SPV-DAG: OpTypeImage %float SubpassData 2 0 0 2 Unknown
// CHECK_GLSL: subpassInput

layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput subpass;

// CHECK_SPV-DAG: InputAttachmentIndex 1
// CHECK_SPV-DAG: OpTypeImage %float SubpassData 2 0 1 2 Unknown
// CHECK_GLSL: subpassInputMS
layout (input_attachment_index = 1, set = 0, binding = 1) uniform subpassInputMS subpassMS;

layout (location = 0) out vec4 outColor;

void someSideEffect(subpassInput subpassTmp)
{
    outColor.xy = subpassLoad(subpassTmp).xy;
}

void someSideEffectMS(subpassInputMS subpassTmp)
{
    outColor.zw = subpassLoad(subpassTmp, 0).zw;
}

// CHECK_GLSL-DAG: void main(
// CHECK_SPV-DAG: OpEntryPoint

void main() {
    someSideEffect(subpass);
    someSideEffectMS(subpassMS);
}