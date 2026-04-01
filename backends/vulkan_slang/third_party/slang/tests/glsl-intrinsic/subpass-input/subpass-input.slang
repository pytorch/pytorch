//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -stage fragment -entry main -allow-glsl

#version 450
// CHECK_SPV-DAG: OpEntryPoint

// CHECK_SPV-DAG: InputAttachmentIndex 0
// CHECK_SPV-DAG: InputAttachmentIndex 1
// CHECK_SPV-DAG: InputAttachmentIndex 2
// CHECK_SPV-DAG: InputAttachmentIndex 3
// CHECK_SPV-DAG: InputAttachmentIndex 4
// CHECK_SPV-DAG: InputAttachmentIndex 5

// CHECK_SPV-DAG: OpTypeImage %float SubpassData 2 0 0 2 Unknown
// CHECK_GLSL-DAG: input_attachment_index = 0
// CHECK_GLSL-DAG: subpassInput
// CHECK_HLSL-DAG: vk::input_attachment_index(0)
// CHECK_HLSL-DAG: SubpassInput<float4>
layout (input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput subpass;

// CHECK_SPV-DAG: OpTypeImage %float SubpassData 2 0 1 2 Unknown
// CHECK_GLSL-DAG: input_attachment_index = 1
// CHECK_GLSL-DAG: subpassInputMS
// CHECK_HLSL-DAG: vk::input_attachment_index(1)
// CHECK_HLSL-DAG: SubpassInputMS<float4>
layout (input_attachment_index = 1, set = 0, binding = 1) uniform subpassInputMS subpassMS;

// CHECK_SPV-DAG: OpTypeImage %uint SubpassData 2 0 0 2 Unknown
// CHECK_GLSL-DAG: input_attachment_index = 2
// CHECK_GLSL-DAG: usubpassInput
// CHECK_HLSL-DAG: vk::input_attachment_index(2)
// CHECK_HLSL-DAG: SubpassInput<uint4>
layout (input_attachment_index = 2, set = 0, binding = 2) uniform usubpassInput usubpass;

// CHECK_SPV-DAG: OpTypeImage %uint SubpassData 2 0 1 2 Unknown
// CHECK_GLSL-DAG: input_attachment_index = 3
// CHECK_GLSL-DAG: usubpassInputMS
// CHECK_HLSL-DAG: vk::input_attachment_index(3)
// CHECK_HLSL-DAG: SubpassInputMS<uint4>
layout (input_attachment_index = 3, set = 0, binding = 3) uniform usubpassInputMS usubpassMS;

// CHECK_SPV-DAG: OpTypeImage %int SubpassData 2 0 0 2 Unknown
// CHECK_GLSL-DAG: input_attachment_index = 4
// CHECK_GLSL-DAG: isubpassInput
// CHECK_HLSL-DAG: vk::input_attachment_index(4)
// CHECK_HLSL-DAG: SubpassInput<int4>
layout (input_attachment_index = 4, set = 0, binding = 4) uniform isubpassInput isubpass;

// CHECK_SPV-DAG: OpTypeImage %int SubpassData 2 0 1 2 Unknown
// CHECK_GLSL-DAG: input_attachment_index = 5
// CHECK_GLSL-DAG: isubpassInputMS
// CHECK_HLSL-DAG: vk::input_attachment_index(5)
// CHECK_HLSL-DAG: SubpassInputMS<int4>
layout (input_attachment_index = 5, set = 0, binding = 5) uniform isubpassInputMS isubpassMS;

layout (location = 0) out vec4 outColor;

// CHECK_GLSL-DAG: void main()
// CHECK_HLSL-DAG: main()
void main() {
// CHECK_SPV: OpImageRead
// CHECK_GLSL: subpassLoad
// CHECK_HLSL: SubpassLoad
// CHECK_SPV: OpImageRead
// CHECK_GLSL: subpassLoad
// CHECK_HLSL: SubpassLoad
// CHECK_SPV: OpImageRead
// CHECK_GLSL: subpassLoad
// CHECK_HLSL: SubpassLoad
// CHECK_SPV: OpImageRead
// CHECK_GLSL: subpassLoad
// CHECK_HLSL: SubpassLoad
// CHECK_SPV: OpImageRead
// CHECK_GLSL: subpassLoad
// CHECK_HLSL: SubpassLoad
// CHECK_SPV: OpImageRead
// CHECK_GLSL: subpassLoad
// CHECK_HLSL: SubpassLoad
    outColor = vec4(true
        && subpassLoad(subpass) == vec4(1)
        && subpassLoad(subpassMS, 0) == vec4(1)
        && subpassLoad(isubpass) == ivec4(1)
        && subpassLoad(isubpassMS, 0) == ivec4(1)
        && subpassLoad(usubpass) == uvec4(1)
        && subpassLoad(usubpassMS, 0) == uvec4(1)
    );
}