//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-slang -shaderobj -mtl

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

static uint g_values[2] = { 0, 1 };

[numthreads(2, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	outputBuffer[tid] = g_values[tid];
    // BUF: 0
    // BUF: 1
}
