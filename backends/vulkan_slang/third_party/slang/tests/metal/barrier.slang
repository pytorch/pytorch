//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib

// CHECK: threadgroup_barrier

// CHECK-ASM: define void @computeMain

RWStructuredBuffer<float> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    outputBuffer[0] = 1;
    GroupMemoryBarrierWithGroupSync();
    outputBuffer[1] = 2;
    AllMemoryBarrierWithGroupSync();
    outputBuffer[2] = 3;
}