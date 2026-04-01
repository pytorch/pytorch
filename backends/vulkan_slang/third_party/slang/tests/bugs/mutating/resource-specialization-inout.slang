// Bug related to resource specialization on unused resource typed fields.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

//TEST_INPUT: Texture2D(size=4, content = one):name t2D
Texture2D t2D;

struct MyType
{
    Texture2D tex;
    int data;
    [mutating]
    void f(int val)
    {
        data += val;
    }
}

int test(int val)
{
    MyType t = {t2D, 0};
    t.f(val);
    return t.data;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=gOutputBuffer
RWStructuredBuffer<int> gOutputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inputVal = tid;
    int outputVal = test(inputVal);
    gOutputBuffer[tid] = outputVal;
}
