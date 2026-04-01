//TEST:SIMPLE(filecheck=CHECK_ERROR): -target spirv -entry psmain -profile vs_6_0
//TEST:SIMPLE(filecheck=CHECK_ERROR): -target spirv -entry vsmain -profile ps_6_0

//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry vsmain -profile vs_6_0
//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry psmain -profile vs_6_0 -ignore-capabilities
//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry vsmain -profile ps_6_0 -ignore-capabilities -skip-spirv-validation

// CHECK_ERROR: warning 36112
// CHECK-NOT: warning 36112

struct CameraProperties {
    float4x4 MVP;
};

[[vk::push_constant]]
ConstantBuffer<CameraProperties> Cam;

struct VSOutput {
    float4 PositionCS : SV_POSITION;
    float3 Color      : COLOR;
};

[shader("vertex")]
VSOutput vsmain(float3 PositionOS : POSITION, float3 Color : COLOR0)
{
    VSOutput output = (VSOutput)0;
    output.PositionCS = mul(Cam.MVP, float4(PositionOS, 1));
    output.Color = Color;
    return output;
}

[shader("pixel")]
float4 psmain(VSOutput input) : SV_TARGET
{
    return float4(input.Color, 1);   
}
