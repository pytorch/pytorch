// static-const-in-struct.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that `static const` variable declarations inside of
// a `struct` type correctly translate to constants in
// the output.

struct Test
{
    static const int kArraySize = 4;

    // First case: test that a `static const` can
    // be used as a constant inside the scope of
    // methods on the type.
    //
    int method(int val)
    {
        int a[kArraySize] = {};

        for(int i = 0; i < kArraySize; ++i)
            a[i] = i;

        return val * 16 + a[val];
    }
}

// Second case: test that a global function can
// reference and use a constant scoped within
// a structure type.
//
int globalFunc(int val)
{
    int a[Test.kArraySize] = {};

    for(int i = 0; i < Test::kArraySize; ++i)
        a[i] = i;

    return val * 16 + a[val];
}

int test(int val)
{
    Test t;
    return t.method(val) * 256 + globalFunc(val);
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
