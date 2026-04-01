// glsl-bool-ops.slang

// This test exists to ensure that binary operations on
// vectors of `bool` produce correct GLSL/SPIR-V output.

//TEST(compute):COMPARE_COMPUTE:-dx11 -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE:-vk -compute -shaderobj

uint test(uint val)
{
	bool2 a = val > uint2(0);
	bool2 b = (val & uint2(1)) == 0;
	bool2 c = val < uint2(3);

	bool2 d = a && b;
	bool2 e = a || c;

	bool2 f = !(a & b);
	bool2 g = a | c;
	bool2 h = a ^ c;

	uint2 t = 0;
	t = t*2 + d;
	t = t*2 + e;
	t = t*2 + f;
	t = t*2 + g;
	t = t*2 + h;

	uint result = 0;
	result = result*256 + t.x;
	result = result*256 + t.y;
	return result;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<uint> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    outputBuffer[tid] = test(tid);
}
