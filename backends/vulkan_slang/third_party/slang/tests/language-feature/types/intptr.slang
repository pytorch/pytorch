// intptr.slang

// Test pointer-sized integer types.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -cpu -compute

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

__target_intrinsic(cpp, "sizeof(intptr_t)")
__target_intrinsic(cuda, "sizeof(size_t)")
intptr_t getNativeIntPtrSize();

__generic<T>
__target_intrinsic(cpp, "sizeof($0)")
__target_intrinsic(cuda, "sizeof($0)")
intptr_t getSizeOf(T val);

intptr_t getIntPtrVal() { return 0z; }

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    let expectedSize = getNativeIntPtrSize();
    intptr_t a = 0z;
    var actualSize = getSizeOf(getIntPtrVal());
    int testResult = 1;
    if (expectedSize != actualSize)
    {
        testResult = 0;
    }
    uintptr_t b = 0uz;
    actualSize = getSizeOf(b);
    if (expectedSize != actualSize)
    {
        testResult = 0;
    }
    outputBuffer[dispatchThreadID.x] = testResult;
}
