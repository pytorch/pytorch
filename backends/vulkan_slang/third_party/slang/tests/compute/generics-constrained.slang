//TEST(smoke,compute):COMPARE_COMPUTE: -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj

// Use interface constraints on a generic parameter

interface Helper
{
	float getHelp();	
}

struct A : Helper
{
	float a;

	float getHelp()
	{
		// TODO: we should be able to reference a member variable here,
		// but the front-end isn't handling references through `this`
		// properly yet.
		return a;
	}	
};

__generic<T : Helper>
float testHelp(T helper)
{
	return helper.getHelp();
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	float inVal = float(tid);

	A a;
	a.a = inVal;
	float outVal = testHelp<A>(a);

	outputBuffer[tid] = outVal;
}