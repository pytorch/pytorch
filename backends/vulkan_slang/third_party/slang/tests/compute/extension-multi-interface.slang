//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IAdd
{
    float addf(float u, float v);
}

interface ISub
{
    float subf(float u, float v);
}

interface IAddAndSub
{
    float addf(float u, float v);
    float subf(float u, float v);    
}

struct Simple : IAdd
{
    float base;
    float addf(float u, float v)
    {
        return u+v;
    }
};

__extension Simple : ISub, IAddAndSub
{
    float subf(float u, float v)
    {
        return base+u-v;
    }
};

float testAddSub<T:IAddAndSub>(T t)
{
    return t.subf(t.addf(1.0, 1.0), 1.0);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Simple s;
    s.base = 0.0;
	float outVal = testAddSub(s);
	outputBuffer[dispatchThreadID.x] = outVal;
}