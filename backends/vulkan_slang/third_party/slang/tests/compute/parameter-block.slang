//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj
//TEST(compute):COMPARE_COMPUTE:-shaderobj
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl -render-features argument-buffer-tier-2
//TEST(compute):COMPARE_COMPUTE:-wgpu

// Ensure that Slang `ParameterBlock` type is lowered
// to HLSL in the fashion that we expect.

struct P
{
	RWStructuredBuffer<int> buffer;
};

//TEST_INPUT: set block0 = new{ out ubuffer(data=[0 0 0 0], stride=4) }
ParameterBlock<P> block0;

//TEST_INPUT: set block1 = new{ ubuffer(data=[0 1 2 3], stride=4) }
ParameterBlock<P> block1;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	int inVal = block1.buffer[tid];

	int outVal = inVal;

	block0.buffer[tid] = outVal;
}
