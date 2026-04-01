//TEST:SIMPLE(filecheck=CHECK_SPIRV): -target spirv -fvk-use-entrypoint-name
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -fvk-use-entrypoint-name

// 
// Check that SPIRV quad control execution modes and GLSL layout/attribute decorations are only 
// set on entry points that contain quad control functions and/or quad control decorations.
//

Texture2D colorTexture1;
SamplerState samplerState;

struct FragmentInput {
    float2 uv : TEXCOORD0;
};

float4 getFragColor(float2 uv) {
    float4 fragColor = float4(1.0, 1.0, 1.0, 1.0);
    bool nonUniformCondition = uv.x > 0.5;

    if (QuadAny(nonUniformCondition)) {
        float4 color = colorTexture1.Sample(samplerState, uv);
        if (nonUniformCondition) {
            fragColor = color;
        }
    }

    return fragColor;
}

// CHECK_SPIRV: OpExtension "SPV_KHR_maximal_reconvergence"
// CHECK_SPIRV: OpExecutionMode %fragmentMain1 MaximallyReconvergesKHR
// CHECK_SPIRV: OpExecutionMode %fragmentMain1 QuadDerivativesKHR
// CHECK_GLSL: layout(quad_derivatives) in
// CHECK_GLSL: [maximally_reconverges]
[shader("fragment")]
float4 fragmentMain1(FragmentInput input) : SV_Target
{
    bool nonUniformCondition = input.uv.x > 0.5;

    float4 fragColor = float4(1.0, 1.0, 1.0, 1.0);

    if (QuadAny(nonUniformCondition)) {
        float4 color = colorTexture1.Sample(samplerState, input.uv);
        if (nonUniformCondition) {
            fragColor = color;
        }
    }

    return fragColor;
}

// CHECK_SPIRV-NOT: OpExecutionMode %fragmentMain2 QuadDerivativesKHR
// CHECK_SPIRV: OpExecutionMode %fragmentMain2 MaximallyReconvergesKHR
// CHECK_GLSL-NOT: layout(quad_derivatives) in
// CHECK_GLSL: [maximally_reconverges]
[MaximallyReconverges]
[shader("fragment")]
float4 fragmentMain2(FragmentInput input) : SV_Target
{
    return float4(1.0, 1.0, 1.0, 1.0);
}


// CHECK_SPIRV-NOT: OpExecutionMode %fragmentMain3 MaximallyReconvergesKHR
// CHECK_SPIRV-NOT: OpExecutionMode %fragmentMain3 QuadDerivativesKHR
// CHECK_SPIRV: OpExecutionMode %fragmentMain3 RequireFullQuadsKHR
// CHECK_GLSL-NOT: layout(quad_derivatives) in
// CHECK_GLSL: layout(full_quads) in
// CHECK_GLSL-NOT: [maximally_reconverges]
[RequireFullQuads]
[shader("fragment")]
float4 fragmentMain3(FragmentInput input) : SV_Target
{
    return float4(1.0, 1.0, 1.0, 1.0);
}

// CHECK_SPIRV: OpExecutionMode %fragmentMain4 MaximallyReconvergesKHR
// CHECK_SPIRV: OpExecutionMode %fragmentMain4 QuadDerivativesKHR
// CHECK_GLSL: layout(quad_derivatives) in
// CHECK_GLSL: [maximally_reconverges]
[shader("fragment")]
float4 fragmentMain4(FragmentInput input) : SV_Target
{
    return getFragColor(input.uv);
}

// CHECK_SPIRV-NOT: OpExecutionMode %fragmentMain5 MaximallyReconvergesKHR
// CHECK_SPIRV-NOT: OpExecutionMode %fragmentMain5 QuadDerivativesKHR
// CHECK_SPIRV-NOT: OpExecutionMode %fragmentMain5 RequireFullQuadsKHR
// CHECK_GLSL-NOT: layout(quad_derivatives) in
// CHECK_GLSL-NOT: layout(full_quads) in
// CHECK_GLSL-NOT: [maximally_reconverges]
[shader("fragment")]
float4 fragmentMain5(FragmentInput input) : SV_Target
{
    return float4(1.0, 1.0, 1.0, 1.0);
}
