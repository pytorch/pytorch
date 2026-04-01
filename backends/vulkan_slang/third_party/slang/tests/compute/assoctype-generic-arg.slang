//DISABLED_TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//DISABLED_TEST(compute):COMPARE_COMPUTE: -shaderobj

//TEST_INPUT:type AssocImpl

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IBase
{
    float getVal();
};

interface IAssoc
{
    associatedtype TBase : IBase;
};

struct BaseImpl : IBase
{
    float getVal() { return 1.0; }
};

struct AssocImpl : IAssoc
{
    typedef BaseImpl TBase;
};

[numthreads(4, 1, 1)]
void computeMain<T : IAssoc>(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    T.TBase base;
    float rs = base.getVal();
    outputBuffer[tid] = rs;
}