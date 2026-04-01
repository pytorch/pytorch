//TEST:SIMPLE(filecheck=CHECK): -target hlsl -profile cs_5_0 -entry computeMain -line-directive-mode none

// Test that we can DCE load of a rw buffer.

RWStructuredBuffer<float> gOutputBuffer;

float test()
{
    return gOutputBuffer[0];
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    test();
}

// CHECK: void computeMain
// CHECK-NOT: test
// CHECK: }

