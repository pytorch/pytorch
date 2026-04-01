//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -dx12 -use-dxil -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cuda -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -metal -output-using-type
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -entry computeMain -stage compute

// METALLIB: computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:set bbuffer = ubuffer(data=[0 0 0 0])
RWByteAddressBuffer bbuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint originalValue;
    bbuffer.InterlockedAdd(0, 1);

    bbuffer.InterlockedAdd(0, 1, originalValue);
    outputBuffer[4] = originalValue;

    bbuffer.InterlockedMax(0, 3);
    bbuffer.InterlockedMax(0, 4, originalValue);
    outputBuffer[5] = originalValue;

    bbuffer.InterlockedMin(0, 2);
    bbuffer.InterlockedMin(0, 2, originalValue);
    outputBuffer[6] = originalValue;

    bbuffer.InterlockedOr(0, 1);
    bbuffer.InterlockedOr(0, 1, originalValue);
    outputBuffer[7] = originalValue;

    bbuffer.InterlockedXor(0, 4);
    bbuffer.InterlockedXor(0, 4, originalValue);
    outputBuffer[8] = originalValue;

    bbuffer.InterlockedAnd(0, 7);
    bbuffer.InterlockedAnd(0, 7, originalValue);
    outputBuffer[9] = originalValue;

    bbuffer.InterlockedCompareExchange(4, 0, 1, originalValue);
    outputBuffer[10] = originalValue;

    bbuffer.InterlockedExchange(8, 3, originalValue);
    outputBuffer[11] = originalValue;

    bbuffer.InterlockedCompareStore(12, 0, 3);

    // CHECK: 3
    // CHECK: 1
    // CHECK: 3
    // CHECK: 3
    outputBuffer[0] = bbuffer.Load(0);
    outputBuffer[1] = bbuffer.Load(4);
    outputBuffer[2] = bbuffer.Load(8);
    outputBuffer[3] = bbuffer.Load(12);
    // CHECK: 1
    // CHECK: 3
    // CHECK: 2
    // CHECK: 3
    // CHECK: 7
    // CHECK: 3
    // CHECK: 0
    // CHECK: 0
}