// frem.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj -emit-spirv-directly -output-using-type

// Test uses of floating-point `%` operator.

RWStructuredBuffer<float> gInput;
//TEST_INPUT:ubuffer(data=[2.0 1.0 5.0 2.0 1.0 -4.0 -5.0 4.0], stride=4):name=gInput

int test(int inVal)
{
    float a = gInput[inVal*2];
    float b = gInput[inVal*2 + 1];
    return int(a % b);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[Shader("compute")]
[NumThreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	int inVal = tid;
	int outVal = test(inVal);
	outputBuffer[tid] = outVal;
}
