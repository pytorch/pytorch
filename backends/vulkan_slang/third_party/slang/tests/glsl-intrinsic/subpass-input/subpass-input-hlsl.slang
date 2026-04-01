//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -stage fragment -entry main -allow-glsl

// CHECK_SPV-DAG: OpEntryPoint
// CHECK_SPV-DAG: InputAttachmentIndex 3
// CHECK_SPV-DAG: InputAttachmentIndex

// CHECK_GLSL-DAG: void main()
// CHECK_GLSL-DAG: input_attachment_index = 3
// CHECK_GLSL-DAG: input_attachment_index

// CHECK_HLSL-DAG: main()
// CHECK_HLSL-DAG: [vk::input_attachment_index(3)]
// CHECK_HLSL-DAG: vk::input_attachment_index
[[vk::input_attachment_index(3)]] SubpassInput subpassHLSL1;
SubpassInput subpassHLSL2;

RWTexture2D<float4> t;

layout (location = 0) out vec4 outColor;

void main() {
// CHECK_SPV-COUNT-2: OpImageRead
// CHECK_GLSL-COUNT-2: subpassLoad
// CHECK_HLSL-COUNT-2: SubpassLoad
    outColor = vec4(true
        && subpassHLSL1.SubpassLoad() == vec4(1)
        && subpassHLSL2.SubpassLoad() == vec4(1)
        && t.Load(int2(0,0)) != float4(0,0,0,0)
    );
}