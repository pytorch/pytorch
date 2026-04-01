// constant-fold.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

struct AnotherStruct<let SomeValue : int>
{
    __init()
    {
        value = SomeValue;
    }
    int value;
};

struct SomeStruct<let SomeValue : bool>
{
    __init()
    {
        value = SomeValue;
    }
    bool value;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    SomeStruct<true> a;
    SomeStruct<false> b;
    SomeStruct<bool(1)> c;
    SomeStruct<bool(0)> d;
    
    AnotherStruct<true> e;
    AnotherStruct<false> f;
    AnotherStruct<2> g;
    AnotherStruct<-2> h;
    
    
    uint tid = dispatchThreadID.x;
    
    int outVal;
    switch (tid)
    {
        default:
        case 0: outVal = a.value; break;
        case 1: outVal = b.value; break;
        case 2: outVal = c.value; break;
        case 3: outVal = d.value; break;
        case 4: outVal = e.value; break;
        case 5: outVal = f.value; break;
        case 6: outVal = g.value; break;
        case 7: outVal = h.value; break;
    }
    
    outputBuffer[tid] = outVal;
}
