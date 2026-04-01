//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj


// Access fields of a `struct` type from within a "method" by
// using an explicit `this` expression.

struct A
{
	float x;

	float addWith(float y)
	{
		return this.x + y;
	}
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;


float test(float inVal)
{
	A a;
	a.x = inVal;
	return a.addWith(inVal*inVal);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;
	float inVal = float(tid);
	float outVal = test(inVal);
	outputBuffer[tid] = outVal;
}