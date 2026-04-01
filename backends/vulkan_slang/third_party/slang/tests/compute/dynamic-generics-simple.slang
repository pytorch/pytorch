//TEST(compute):COMPARE_COMPUTE:-cpu -xslang -disable-specialization
//DISABLE_TEST(compute):COMPARE_COMPUTE:-cuda -xslang -disable-specialization

// Test basic dynamic dispatch code gen

[anyValueSize(16)]
interface IInterface
{
	static int Compute(int inVal);
};

int GenericCompute<T:IInterface>(int inVal)
{
	return T.Compute(inVal);
}

struct Impl : IInterface
{
	static int Compute(int inVal) { return inVal * inVal; }
};

int test(int inVal)
{
	return GenericCompute<Impl>(inVal);
}

//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer : register(u0);

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	int inVal = outputBuffer[tid];
	int outVal = test(inVal);
	outputBuffer[tid] = outVal;
}
