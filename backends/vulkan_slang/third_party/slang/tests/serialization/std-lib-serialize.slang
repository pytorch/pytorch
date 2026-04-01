//TEST:COMPILE: -save-core-module slang-core-module.zip
//TEST:COMPARE_COMPUTE: -compile-arg -load-core-module -compile-arg slang-core-module.zip -shaderobj

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
