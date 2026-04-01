//TEST(compute):COMPARE_COMPUTE:-dx11 -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -xslang -disable-specialization -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -xslang -disable-specialization -shaderobj

// Test dynamic dispatch code gen for generic-typed local variables.

[anyValueSize(8)]
interface IInterface
{
    [mutating]
    int Compute(int inVal);
};

int GenericCompute<T:IInterface>(int inVal)
{
    T obj;
    obj.Compute(3);
    return obj.Compute(inVal); // 3 + inVal
}

struct Impl : IInterface
{
    int base;
    [mutating]
    int Compute(int inVal)
    {
        int result = base + inVal;
        base = inVal;
        return result;
    }
};

int test(int inVal)
{
    return GenericCompute<Impl>(1);
}

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    int inVal = outputBuffer[tid];
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
