//TEST(compute):COMPARE_COMPUTE:-dx11 -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -xslang -disable-specialization -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -xslang -disable-specialization -shaderobj

// Test dynamic dispatch code gen for specializing a generic with
// an existential value.

[anyValueSize(16)]
interface IInterface
{
	int Compute(int inVal);
};

int GenericCompute0(IInterface obj, int inVal)
{
    return GenericCompute1(obj, obj, inVal);
}

int GenericCompute1<T:IInterface>(T obj, IInterface obj2, int inVal)
{
    return obj.Compute(inVal) + obj2.Compute(inVal);
}


struct Impl : IInterface
{
    int base;
	int Compute(int inVal) { return base + inVal * inVal; }
};

int test(int inVal)
{
    Impl obj;
    obj.base = 1;
	return GenericCompute0(obj, inVal);
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
