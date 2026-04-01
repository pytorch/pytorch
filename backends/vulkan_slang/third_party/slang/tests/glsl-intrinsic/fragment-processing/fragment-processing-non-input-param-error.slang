//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage fragment -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_SPV): -target spirv -emit-spirv-directly -stage fragment -entry main -allow-glsl
#version 450
// CHECK_SPV-DAG: error 31208
// CHECK_GLSL-DAG: error 31208

layout(location = 0) out ivec4 outColorActual;

layout (location = 0) in float inDataV1;
bool testFragmentProcessingInterpolateFunctions()
{
    float v = 1.0f;
    return interpolateAtCentroid(v) != -1.0f;
}

void main() {
    outColorActual = ivec4(true
        && testFragmentProcessingInterpolateFunctions()
        );
}