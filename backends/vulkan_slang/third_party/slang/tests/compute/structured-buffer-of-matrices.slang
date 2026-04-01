// structured-buffer-of-matrices.slang

// This test confirms that we have implemented a workaround
// for the issue where the Microsoft HLSL compilers (fxc and dxc)
// ignore matrix layout settings (like `#pragma pack_matrx`)
// in the case where a matrix type is used directly in a
// structured buffer (e.g., `StructuredBuffer<float4x4>`),
// and instead always use column-major layout.
//
// With our fix in place, this test should produce the same
// output on all targets, correctly respecting the user's
// request for row-major layout via command-line option.

//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -compile-arg -O3 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -xslang -matrix-layout-row-major -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -xslang -matrix-layout-row-major -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -xslang -matrix-layout-row-major -shaderobj


// Note: we are using a buffer of floating-point matrices, but fill it with integer
// data, to allow this test to work on the Vulkan targets, which do not currently
// support integer matrices, because of GLSL limitations.

//TEST_INPUT:ubuffer(data=[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15], stride=64):name gInput
RWStructuredBuffer<float4x4> gInput;


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gOutput
RWStructuredBuffer<int> gOutput;

int test(int val)
{
	return asint(gInput[0][(val+1)&3][(val+3)&3]);
}

[numthreads(4, 1, 1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
	int value = tid.x;
	value = test(value);
	gOutput[tid.x] = value;
}
