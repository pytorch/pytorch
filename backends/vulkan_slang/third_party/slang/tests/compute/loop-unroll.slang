//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-dx12  -shaderobj
//TEST(compute):COMPARE_COMPUTE:-dx12 -use-dxil -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cuda -shaderobj
// Note VK output is not loop unrolled
//TEST(compute):COMPARE_COMPUTE:-vk -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Arrays of textures or buffers
//DISABLE_TEST(compute):COMPARE_COMPUTE:-wgpu

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name buffers[0]
//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):name buffers[1]
//TEST_INPUT:ubuffer(data=[1 2 3 0], stride=4):name buffers[2]

// Check that we propagate the `[unroll]` attribute
// through to HLSL output correctly.
//
// If we neglect to generate the attribute in the output,
// it will generate a warning output from fxc, and the
// test will fail to match the expected output.

RWStructuredBuffer<int> buffers[3];

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	// Note: using `unroll` as a variable name to validate that
	// the lookup process for attribute names doesn't run into
	// problems because of local declarations with the same name.
	int unroll = buffers[2][tid];

	[unroll]
	for(int ii = 0; ii < 2; ii++)
	{
		unroll = buffers[ii + 1][unroll];
	}

	buffers[0][tid] = unroll;
}
