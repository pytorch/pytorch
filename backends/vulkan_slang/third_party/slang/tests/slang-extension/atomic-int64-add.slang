//TEST:SIMPLE(filecheck=CHECK_HLSL): -stage compute -entry computeMain -target hlsl
//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage compute -entry computeMain -target glsl

RWByteAddressBuffer outputBuffer;

// CHECK_GLSL: atomicAdd
// CHECK_GLSL: atomicAdd
// CHECK_HLSL: InterlockedAdd64
// CHECK_HLSL: InterlockedAdd64

[numthreads(16, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    uint64_t indata1;
    int64_t indata2;
    outputBuffer.InterlockedAdd64(0, 1, indata1);
    outputBuffer.InterlockedAdd64(8, 1, indata2);
}