// atomics-buffer.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
// Doesn't work on VK - GLSL output doesn't replace InterlockedAdd.  
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -vk -shaderobj
// Cannot work on CUDA, as outputBuffer becomes a CUsurfObject - which do not appear to have atomics available.
// If the buffer was a StructuredBuffer this would work on CUDA.
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cuda -shaderobj
// Atomics not available on CPU currently
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj
// RWBuffer does not work with the GFX backend as expected with Metal
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
//TEST:SIMPLE(filecheck=METALLIB): -target metallib -stage compute -entry computeMain
// Not supported in WGSL: Use of traditional atomics intrinsics (InterlockedXXX functions)
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -wgpu

//METALLIB: @computeMain

//TEST_INPUT:ubuffer(format=R_UInt32, data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]):out,name outputBuffer

RWBuffer<uint> outputBuffer;

void test(uint val)
{
    uint originalValue;

	InterlockedAdd(outputBuffer[val], 		val, 		originalValue);
	InterlockedAdd(outputBuffer[val ^ 1], 	val*16, 	originalValue);
	InterlockedAdd(outputBuffer[val ^ 2], 	val*16*16, 	originalValue);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;
    test(tid);
}
