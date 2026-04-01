// legalize-struct-init.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
//TEST_INPUT:ubuffer(data=[4 3 2 1], stride=4):name inputBuffer


RWStructuredBuffer<uint> outputBuffer;
RWStructuredBuffer<uint> inputBuffer;

struct Helper
{
    RWStructuredBuffer<uint> o;
    RWStructuredBuffer<uint> i;
    uint                    t;  
};

void test(Helper h)
{
    h.o[h.t] = h.i[h.t] * 16 + h.t;    
}

void test(uint t)
{
    Helper h = { outputBuffer, inputBuffer, t };
    test(h);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    test(tid);
}
