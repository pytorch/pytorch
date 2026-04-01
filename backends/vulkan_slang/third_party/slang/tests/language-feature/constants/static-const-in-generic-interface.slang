// static-const-in-generic-interface.slang

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj

// Test that `static const` variable declarations inside of
// a generic `interface` type correctly translate to interface requirements.

interface ITest<T:__BuiltinIntegerType>
{
    static const T kUserDefinedValue;
}

struct Impl : ITest<int>
{
    static const int kUserDefinedValue = 4;
}

struct EnsureCompileTimeEval<T : __BuiltinIntegerType>
{
    static T getValue<U : ITest<T>>() { return U.kUserDefinedValue; }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    static const int result = EnsureCompileTimeEval<int>.getValue<Impl>();
    int outVal = result;
    // CHECK: 4
    outputBuffer[0] = outVal;
}
