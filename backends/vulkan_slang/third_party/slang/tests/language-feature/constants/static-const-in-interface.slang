// static-const-in-interface.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that `static const` variable declarations inside of
// a `interface` type correctly translate to interface requirements.

interface ITest
{
    static const int kUserDefinedValue;
}

extension int : ITest
{
    static const int kUserDefinedValue = 4;
}

extension int16_t : ITest
{
    static const int kUserDefinedValue = 2;
}

struct GetValue<T : ITest>
{
    static const int value = T.kUserDefinedValue;
}

struct EnsureCompileTimeEval<let u : int>
{
    static const int value = u;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    static const int result = EnsureCompileTimeEval<GetValue<int>.value - GetValue<int16_t>.value>.value;
    int outVal = result;
    outputBuffer[tid] = outVal;
}
