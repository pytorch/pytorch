//TEST(compute):COMPARE_COMPUTE: -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IBase
{
    float get();
}

struct BaseImpl : IBase
{
    float get() { return 1.0; }
};

__generic<T:IBase>
float eval(T obj)
{
    return obj.get();
}

__generic<T : IBase>
float test(T obj)
{
    return eval<T>(obj);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    BaseImpl base;
	float outVal = test<BaseImpl>(base);
	outputBuffer[dispatchThreadID.x] = outVal;
}