//TEST:SIMPLE(filecheck=CHECK_SPIRV): -entry fragmentMain -stage fragment -target spirv
//TEST:SIMPLE(filecheck=CHECK_GLSL): -entry fragmentMain -stage fragment -target glsl
//TEST:SIMPLE(filecheck=CHECK_HLSL): -entry fragmentMain -stage fragment -target hlsl
//TEST:SIMPLE(filecheck=CHECK_METAL): -entry fragmentMain -stage fragment -target metal

Texture2D colorTexture1;
Texture2D colorTexture2;
SamplerState samplerState;

struct FragmentInput {
    float2 uv : TEXCOORD0;
};

// CHECK_SPIRV: OpExecutionMode %fragmentMain MaximallyReconvergesKHR
// CHECK_SPIRV: OpExecutionMode %fragmentMain QuadDerivativesKHR
// CHECK_SPIRV: OpExecutionMode %fragmentMain RequireFullQuadsKHR
// CHECK_GLSL: layout(quad_derivatives) in
// CHECK_GLSL: layout(full_quads) in
// CHECK_GLSL: [maximally_reconverges]
[QuadDerivatives]
[RequireFullQuads]
float4 fragmentMain(FragmentInput input) : SV_Target
{
    bool nonUniformCondition1 = input.uv.x > 0.5;
    bool nonUniformCondition2 = input.uv.y > 0.8;

    float4 fragColor = float4(1.0, 1.0, 1.0, 1.0);

    // CHECK_SPIRV: OpGroupNonUniformQuadAnyKHR
    // CHECK_GLSL: subgroupQuadAny
    // CHECK_HLSL: QuadAny
    // CHECK_METAL: quad_any
    if (QuadAny(nonUniformCondition1)) {
        float4 color = colorTexture1.Sample(samplerState, input.uv);
        if (nonUniformCondition1) {
            fragColor = color;
        }
    }

    // CHECK_SPIRV: OpGroupNonUniformQuadAllKHR
    // CHECK_GLSL: subgroupQuadAll
    // CHECK_HLSL: QuadAll
    // CHECK_METAL: quad_all
    if (QuadAll(nonUniformCondition2)) {
        float4 color = colorTexture2.Sample(samplerState, input.uv);
        if (nonUniformCondition2) {
            fragColor += color * 0.5;
        }
    }

    return fragColor;
}

