//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl
#version 450
// CHECK_SPV-DAG: OpEntryPoint
// CHECK_GLSL-DAG: void main(

layout(location = 0) out ivec4 outColorActual;

layout (location = 0) in float inDataV1;

bool isOk(float a, float b)   { return (a == b); }

bool testFragmentProcessingInterpolateFunctions()
{
    float tmpStore = inDataV1;
    return true
        && interpolateAtCentroid(inDataV1) != -1.0f
        && (tmpStore + inDataV1) != -0.1f
        && isOk(inDataV1, tmpStore)
        && inDataV1 != -1.0f
        ;
}

void main() {
    outColorActual = ivec4(true
        && testFragmentProcessingInterpolateFunctions()
        );
}