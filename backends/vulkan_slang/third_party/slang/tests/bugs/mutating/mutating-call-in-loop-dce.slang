//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

// Confirm that a SideEffectFree mutable method does not get DCE'd when
// it is called from within a loop.

struct C
{
    int a;
    [mutating]
    int add()
    {
        a++;
        return a;
    }
    [mutating]
    void init()
    {
        a = 0;
    }
}
int doSomething(C d_in)
{
    C d = d_in;
    int rs = 0;
    for (int i = 0; i < 10; i++)
        rs = d.add();
    return rs;
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out, name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    C c;
    c.init();
    outputBuffer[tid] = doSomething(c);
}
