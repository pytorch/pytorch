// byte-address-buffer.slang

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-d3d12 -compute -shaderobj

// Confirm cross-compilation of `(RW)ByteAddressBuffer`
//
// TODO: I'm only using `RWByteAddressBuffer` for now because I don't
// know if `render-test` supports the non-UAV case.

//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]):name=inputBuffer
RWByteAddressBuffer inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0]):out,name=outputBuffer
RWByteAddressBuffer outputBuffer;

void test(int val)
{
	uint tmp = val;

	tmp = inputBuffer.Load(int(tmp * 4));

	uint2 pair = inputBuffer.Load2(int(tmp * 4));
	tmp = (pair.x + pair.y) & 0xF;

	uint4 quad = inputBuffer.Load4(int(tmp * 4));
	tmp = (quad.x + quad.y + quad.z + quad.w) & 0xF;

	outputBuffer.Store(val*4, tmp);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    int val = int(tid);
    test(val);
}
