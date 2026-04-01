//TEST:SIMPLE(filecheck=CHECK_HLSL): -target hlsl -stage fragment -entry main
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main

bool testFragmentProcessingDerivativeFunctionsScalar()
{
// CHECK_HLSL: ddx
// CHECK_SPV: OpDPdx
// CHECK_GLSL: dFdx
// CHECK_HLSL: ddy
// CHECK_SPV: OpDPdy
// CHECK_GLSL: dFdy
// CHECK_HLSL: ddx_fine
// CHECK_SPV: OpDPdxFine
// CHECK_GLSL: dFdxFine
// CHECK_HLSL: ddy_fine
// CHECK_SPV: OpDPdyFine
// CHECK_GLSL: dFdyFine
// CHECK_HLSL: ddx_coarse
// CHECK_SPV: OpDPdxCoarse
// CHECK_GLSL: dFdxCoarse
// CHECK_HLSL: ddy_coarse
// CHECK_SPV: OpDPdyCoarse
// CHECK_GLSL: dFdyCoarse
// CHECK_HLSL: fwidth
// CHECK_SPV: OpFwidth
// CHECK_GLSL: fwidth
// CHECK_HLSL: abs(ddx_fine({{.*}})) + abs(ddy_fine({{.*}}))
// CHECK_SPV: OpFwidthFine
// CHECK_GLSL: fwidthFine
// CHECK_HLSL: abs(ddx_coarse({{.*}})) + abs(ddy_coarse({{.*}}))
// CHECK_SPV: OpFwidthCoarse
// CHECK_GLSL: fwidthCoarse
    return true
        && ddx(1.0f) != -1.0f
        && ddy(1.0f) != -1.0f
        && ddx_fine(1.0f) != -1.0f
        && ddy_fine(1.0f) != -1.0f
        && ddx_coarse(1.0f) != -1.0f
        && ddy_coarse(1.0f) != -1.0f
        && fwidth(1.0f) != -1.0f
        && fwidth_fine(1.0f) != -1.0f
        && fwidth_coarse(1.0f) != -1.0f
        ;
}
__generic<let N:int>
bool testFragmentProcessingDerivativeFunctionsVector()
{
// CHECK_HLSL: ddx
// CHECK_SPV: OpDPdx
// CHECK_GLSL: dFdx
// CHECK_HLSL: ddy
// CHECK_SPV: OpDPdy
// CHECK_GLSL: dFdy
// CHECK_HLSL: ddx_fine
// CHECK_SPV: OpDPdxFine
// CHECK_GLSL: dFdxFine
// CHECK_HLSL: ddy_fine
// CHECK_SPV: OpDPdyFine
// CHECK_GLSL: dFdyFine
// CHECK_HLSL: ddx_coarse
// CHECK_SPV: OpDPdxCoarse
// CHECK_GLSL: dFdxCoarse
// CHECK_HLSL: ddy_coarse
// CHECK_SPV: OpDPdyCoarse
// CHECK_GLSL: dFdyCoarse
// CHECK_HLSL: fwidth
// CHECK_SPV: OpFwidth
// CHECK_GLSL: fwidth
// CHECK_HLSL: abs(ddx_fine({{.*}})) + abs(ddy_fine({{.*}}))
// CHECK_SPV: OpFwidthFine
// CHECK_GLSL: fwidthFine
// CHECK_HLSL: abs(ddx_coarse({{.*}})) + abs(ddy_coarse({{.*}}))
// CHECK_SPV: OpFwidthCoarse
// CHECK_GLSL: fwidthCoarse
    return true
        && all(ddx(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(ddy(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(ddx_fine(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(ddy_fine(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(ddx_coarse(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(ddy_coarse(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(fwidth(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(fwidth_fine(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        && all(fwidth_coarse(vector<float,N>(1.0f)) != vector<float,N>(-1.0f))
        ;
}

bool testFragmentProcessingFunctions()
{
    return true
        && testFragmentProcessingDerivativeFunctionsScalar()
        && testFragmentProcessingDerivativeFunctionsVector<2>()
        && testFragmentProcessingDerivativeFunctionsVector<3>()
        && testFragmentProcessingDerivativeFunctionsVector<4>()
        ;
    ;
}

[shader("pixel")]
int4 main() : SV_Target {
    return int4(true
        && testFragmentProcessingFunctions()
        );
}
