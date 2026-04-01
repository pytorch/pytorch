//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -stage fragment -entry main
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main

struct VertexOut
{
    float4 pos : SV_Position;
    float3 color;
}

// CHECK_SPV: %v_color = OpVariable %_ptr_Input_v3float Input 
// CHECK_SPV: %{{.*}} = OpExtInst %v3float %{{.*}} InterpolateAtCentroid %v_color
// CHECK_HLSL: EvaluateAttributeAtCentroid(v_0.color_0)

float4 main(VertexOut v) : SV_Target
{
    return float4(EvaluateAttributeAtCentroid(v.color), 1.0);
}