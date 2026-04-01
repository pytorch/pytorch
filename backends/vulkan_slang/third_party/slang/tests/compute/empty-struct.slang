//TEST(smoke,compute):COMPARE_COMPUTE: -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj

// Confirm that generics syntax can be used in user
// code and generates valid output.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct Simple
{
    float getVal() {return 1.0;}
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Simple s;
	float outVal = s.getVal();
	outputBuffer[dispatchThreadID.x] = outVal;
}