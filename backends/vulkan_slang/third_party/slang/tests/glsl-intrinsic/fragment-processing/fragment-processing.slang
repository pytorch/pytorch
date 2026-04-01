//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl
#version 450

layout(location = 0) out ivec4 outColorActual;

layout (location = 0) in float inDataV1;
layout (location = 1) in vec2 inDataV2;
layout (location = 2) in vec3 inDataV3;
layout (location = 3) in vec4 inDataV4;

bool testFragmentProcessingDerivativeFunctionsScalar()
{
// CHECK_SPV: OpDPdx
// CHECK_GLSL: dFdx
// CHECK_SPV: OpDPdy
// CHECK_GLSL: dFdy
// CHECK_SPV: OpDPdxFine
// CHECK_GLSL: dFdxFine
// CHECK_SPV: OpDPdyFine
// CHECK_GLSL: dFdyFine
// CHECK_SPV: OpDPdxCoarse
// CHECK_GLSL: dFdxCoarse
// CHECK_SPV: OpDPdyCoarse
// CHECK_GLSL: dFdyCoarse
// CHECK_SPV: OpFwidth
// CHECK_GLSL: fwidth
// CHECK_SPV: OpFwidthFine
// CHECK_GLSL: fwidthFine
// CHECK_SPV: OpFwidthCoarse
// CHECK_GLSL: fwidthCoarse
    return true
        && dFdx(1.0f) != -1.0f
        && dFdy(1.0f) != -1.0f
        && dFdxFine(1.0f) != -1.0f
        && dFdyFine(1.0f) != -1.0f
        && dFdxCoarse(1.0f) != -1.0f
        && dFdyCoarse(1.0f) != -1.0f
        && fwidth(1.0f) != -1.0f
        && fwidthFine(1.0f) != -1.0f
        && fwidthCoarse(1.0f) != -1.0f
        ;
}
__generic<let N:int>
bool testFragmentProcessingDerivativeFunctionsVector()
{
// CHECK_SPV: OpDPdx
// CHECK_GLSL: dFdx
// CHECK_SPV: OpDPdy
// CHECK_GLSL: dFdy
// CHECK_SPV: OpDPdxFine
// CHECK_GLSL: dFdxFine
// CHECK_SPV: OpDPdyFine
// CHECK_GLSL: dFdyFine
// CHECK_SPV: OpDPdxCoarse
// CHECK_GLSL: dFdxCoarse
// CHECK_SPV: OpDPdyCoarse
// CHECK_GLSL: dFdyCoarse
// CHECK_SPV: OpFwidth
// CHECK_GLSL: fwidth
// CHECK_SPV: OpFwidthFine
// CHECK_GLSL: fwidthFine
// CHECK_SPV: OpFwidthCoarse
// CHECK_GLSL: fwidthCoarse
    return true
        && dFdx(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && dFdy(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && dFdxFine(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && dFdyFine(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && dFdxCoarse(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && dFdyCoarse(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && fwidth(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && fwidthFine(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        && fwidthCoarse(vector<float,N>(1.0f)) != vector<float,N>(-1.0f)
        ;
}
bool testFragmentProcessingInterpolateFunctions()
{
// CHECK_SPV-DAG: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtCentroid %inDataV1
// CHECK_GLSL-DAG: interpolateAtCentroid{{.*}}inDataV1
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtSample %inDataV1 {{.*}}
// CHECK_GLSL: interpolateAtSample{{.*}}inDataV1
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtOffset %inDataV1 {{.*}}
// CHECK_GLSL: interpolateAtOffset{{.*}}inDataV1

// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtCentroid %inDataV2
// CHECK_GLSL: interpolateAtCentroid{{.*}}inDataV2
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtSample %inDataV2 {{.*}}
// CHECK_GLSL: interpolateAtSample{{.*}}inDataV2
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtOffset %inDataV2 {{.*}}
// CHECK_GLSL: interpolateAtOffset{{.*}}inDataV2

// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtCentroid %inDataV3
// CHECK_GLSL: interpolateAtCentroid{{.*}}inDataV3
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtSample %inDataV3 {{.*}}
// CHECK_GLSL: interpolateAtSample{{.*}}inDataV3
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtOffset %inDataV3 {{.*}}
// CHECK_GLSL: interpolateAtOffset{{.*}}inDataV3

// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtCentroid %inDataV4
// CHECK_GLSL: interpolateAtCentroid{{.*}}inDataV4
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtSample %inDataV4 {{.*}}
// CHECK_GLSL: interpolateAtSample{{.*}}inDataV4
// CHECK_SPV: {{.*}} = OpExtInst {{.*}} {{.*}} InterpolateAtOffset %inDataV4 {{.*}}
// CHECK_GLSL: interpolateAtOffset{{.*}}inDataV4
    return true
        && interpolateAtCentroid(inDataV1) != -1.0f
        && interpolateAtSample(inDataV1, 0) != -1.0f
        && interpolateAtOffset(inDataV1, vec2(0.0f)) != -1.0f
        && interpolateAtCentroid(inDataV2) != vector<float,2>(-1.0f)
        && interpolateAtSample(inDataV2, 0) != vector<float,2>(-1.0f)
        && interpolateAtOffset(inDataV2, vec2(0.0f)) != vector<float,2>(-1.0f)
        && interpolateAtCentroid(inDataV3) != vector<float,3>(-1.0f)
        && interpolateAtSample(inDataV3, 0) != vector<float,3>(-1.0f)
        && interpolateAtOffset(inDataV3, vec2(0.0f)) != vector<float,3>(-1.0f)
        && interpolateAtCentroid(inDataV4) != vector<float,4>(-1.0f)
        && interpolateAtSample(inDataV4, 0) != vector<float,4>(-1.0f)
        && interpolateAtOffset(inDataV4, vec2(0.0f)) != vector<float,4>(-1.0f)
        ;
}
bool testFragmentProcessingFunctions()
{
    return true
        && testFragmentProcessingDerivativeFunctionsScalar()
        && testFragmentProcessingDerivativeFunctionsVector<2>()
        && testFragmentProcessingDerivativeFunctionsVector<3>()
        && testFragmentProcessingDerivativeFunctionsVector<4>()
        && testFragmentProcessingInterpolateFunctions()
        ;
    ;
}

void main() {
    outColorActual = ivec4(true
        && testFragmentProcessingFunctions()
        );
}