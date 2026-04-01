// this-type.slang

// Confirm that that `This` type works as expected

//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

interface IFrobable
{
	This frob();
	int getValue();
}

struct Thing : IFrobable
{
	int value;

	Thing frob()
	{
		Thing result = { value * 16 };
		return result;
	}

	int getValue() { return value; }
}

int frobnicate<F:IFrobable>(F f)
{
	return f.frob().getValue();
}

int test(int value)
{
	Thing t = { value };
	return frobnicate(t) + value;
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