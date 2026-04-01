// byte-address-16bit-load.slang

// Test that loading values using 16-bit types from
// byte-address buffers works as expected, on targets
// that support it.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj

// Note: disabled on D3D12 because some of the CI services don't have
// a recent enough build of dxc to support generic load/store.
//
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -profile cs_6_2 -render-features half -shaderobj

//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-slang -vk -compute -profile cs_6_2 -render-features half -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -cuda -compute -shaderobj

// Unavailable on D3D and fxc-based targets, because fxc doesn't
// support loading anything besides `uint` from a byte-address
// buffer.
//
//TEST_DISABLED(compute):COMPARE_COMPUTE_EX:-slang -compute


//TEST_INPUT:ubuffer(data=[0x00010002 0x00030004 0x00050006 0x00070008]):name=inputBuffer
RWByteAddressBuffer inputBuffer;

//TEST_INPUT:ubuffer(data=[0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef]):name=tmpBuffer
RWByteAddressBuffer tmpBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Data
{
	int16_t	a;
	int16_t b;
}

int test(int val)
{
	int offsetX = val*4;
	int offsetY = offsetX & 0xFF;

	Data x = inputBuffer.Load<Data>(offsetX);
	tmpBuffer.Store<Data>(offsetY, x);

	Data y = tmpBuffer.Load<Data>(offsetX);
	return int(y.a)*256*16 + int(y.b);
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	outputBuffer[tid] = test(tid);
}