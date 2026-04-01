// derived-struct-init-list.slang

//TEST(compute):COMPARE_COMPUTE:
//TEST(compute):COMPARE_COMPUTE: -vk -shaderobj

// Test that use of an initializer list (especially
// an empty initializer list) is still possible
// when using `struct` inheritance.

struct Base
{
    int a = 1;
}

struct Derived : Base
{
    int b = 2;

    void write(inout int val) { val = val*0x100 + a*0x10 + b; }
}

int test(int val)
{
    Derived x = {};
    Derived y = { val, val+1 };

    int result = 1;
    x.write(result);
    y.write(result);
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
