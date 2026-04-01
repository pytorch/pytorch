// simple-namespace.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that simple `namespace` declarations work as expected
// Test that the global scope operator `::` works as expected

namespace A
{
    static int num = 16;
    struct X
    {
        int val;

        int getVal()
        {
            return val;
        }
    }
}

namespace B
{
    struct X
    {
        int head;
        int tail;

        int getHead() { return head; }
        int getTail() { return tail; }
    }

    X makeX(int h, int t)
    {
        X result = { h, t };
        return result;
    }
}

namespace A
{
    X makeX(int v)
    {
        X result = { v };
        return result;
    }
}

int test(int val)
{
    A.X a = A::makeX(val);
    // Use the global scope operator "::A::num" to access the static member of namespace A
    // Test it with mul operator
    int num = 16*::A::num;
    B::X b = B.makeX(val*16, val*num);

    return a.getVal() + b.getHead() + b.getTail();
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
