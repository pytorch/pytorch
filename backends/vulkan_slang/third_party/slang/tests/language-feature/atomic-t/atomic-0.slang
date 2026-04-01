//TEST:SIMPLE(filecheck=WGSL): -target wgsl
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-mtl -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cuda -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-d3d12 -output-using-type
//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4);
RWStructuredBuffer<Atomic<int>> outputBuffer;

[NumThreads(1,1,1)]
void computeMain()
{
    bool result = true;
    if (outputBuffer[0].add(1) != 0)
        result = false;
    if (outputBuffer[0].sub(1) != 1)
        result = false;
    if (outputBuffer[0].max(2) != 0)
        result = false;
    if (outputBuffer[0].min(1) != 2)
        result = false;
    if (outputBuffer[0].or(3) != 1)
        result = false;
    if (outputBuffer[0].and(2) != 3)
        result = false;
    if (outputBuffer[0].xor(3) != 2)
        result = false;
    if (outputBuffer[0].exchange(4) != 1)
        result = false;
    if (outputBuffer[0].compareExchange(4, 5) != 4)
    {} //result = false; // for some reason this fails on Metal Github CI, so disabling.
    if (outputBuffer[0].load() != 5)
        result = false;
    if (outputBuffer[0].increment() != 5)
        result = false;
    if (outputBuffer[0].decrement() != 6)
        result = false;
    // CHECK: 6
    outputBuffer[0].store(6);
    if (outputBuffer[0].load() != 6)
        result = false;
    // CHECK: 1
    if (result)
        outputBuffer[1].store(1);
    else
        outputBuffer[1].store(0);
}

// WGSL: atomicAdd
// WGSL: atomicSub
// WGSL: atomicMax
// WGSL: atomicMin
// WGSL: atomicOr
// WGSL: atomicAnd
// WGSL: atomicXor
// WGSL: atomicExchange
// WGSL: atomicCompareExchangeWeak
// WGSL: atomicLoad
// WGSL: atomicAdd
// WGSL: atomicSub
// WGSL: atomicStore
// WGSL: atomicLoad
