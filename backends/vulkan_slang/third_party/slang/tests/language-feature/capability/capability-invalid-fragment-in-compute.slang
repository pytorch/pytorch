//TEST:SIMPLE(filecheck=CHECK): -target hlsl -emit-spirv-directly -entry computeMain -stage compute -allow-glsl
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -emit-spirv-directly -entry computeMain -stage compute -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target hlsl -emit-spirv-directly -entry computeMain -stage compute -allow-glsl -ignore-capabilities
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target hlsl -emit-spirv-directly -entry computeMain -stage compute -allow-glsl -ignore-capabilities

// CHECK_IGNORE_CAPS-NOT: error 36107
// CHECK: error 36107
SamplerState samplerState;
Texture2D<int> rw;

[numthreads(1,1,1)]
void computeMain()
{
    clip(0.0f); // clip is not supported in compute shader, so this throws an error.
}