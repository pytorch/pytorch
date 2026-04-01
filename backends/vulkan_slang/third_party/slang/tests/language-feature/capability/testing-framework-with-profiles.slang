
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -profile sm_6_1
//TEST:SIMPLE(filecheck=CHECK): -target hlsl  -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK): -target hlsl  -entry computeMain -stage compute -capability sm_6_1
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry computeMain -stage compute -profile sm_6_1
//TEST:SIMPLE(filecheck=CHECK_WARN): -target hlsl -entry computeMain -stage compute -capability sm_4_0 

//CHECK-NOT: warning 41012
//CHECK_WARN: warning 41012
//CHECK: computeMain

//TEST_INPUT:ubuffer(data=[1], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // BUF: 1
    outputBuffer[0] = WaveMaskSum(0xFF, 1);
}
