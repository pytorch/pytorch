// property-decl.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that users can declare properties and access them
// with ordinary dot syntax.

struct Test
{
    int storage;

    property value : int
    {
        get { return storage; }
        set(newValue) { storage = newValue; }
    }
}

int test(int val)
{
    Test t;
    t.storage = 0;

    // setter
    t.value = val ^ 0x100;

    // getter + setter
    t.value ^= 0x1;

    // getter
    int result = t.value;

    return result;
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
