//DISABLED_TEST(smoke,compute):COMPARE_COMPUTE: -shaderobj

//TEST_INPUT:type Wrapper<Impl>

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IBase
{
    float compute();
}

struct Wrapper<T : IBase> : IBase
{
    T obj;
    float compute()
    {
        return obj.compute();        
    }
};

struct Impl : IBase
{
    float compute()
    {
        return 1.0;
    }
};

[numthreads(1, 1, 1)]
void computeMain<TImpl:IBase>(
    uniform TImpl impl,
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	float outVal = impl.compute();
	outputBuffer[tid] = outVal;
}