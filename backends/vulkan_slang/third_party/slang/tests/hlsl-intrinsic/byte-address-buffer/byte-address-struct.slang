// byte-address-struct-load.slang

// Test that load of a `struct` type from a byte-address
// buffer can be translated into per-field loads on
// all targets that support loads of the field types.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -use-dxil -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-slang -vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -cuda -compute -shaderobj

// Note: This input should really be just a `ByteAddressBuffer`,
// so that we can confirm that the functionality works in the
// read-only case as well. The reason we are using a `RWByteAddressBuffer`
// is because of limitations in the `render-test` app, and if
// those limitations get fixed, we should switch this code.
//
//TEST_INPUT:ubuffer(data=[0x5 0x3f800000 0x6 0x40000000 0x7 0x40400000 0x8 0x40800000]):name=inputBuffer
RWByteAddressBuffer inputBuffer;

//TEST_INPUT:ubuffer(data=[0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef 0xdeadbeef]):name=tmpBuffer
RWByteAddressBuffer tmpBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Data
{
	int a;
	float b;
}

int test(int val)
{
	int offsetX = val*8;
	int offsetY = offsetX & 0xFF;

	Data x = inputBuffer.Load<Data>(offsetX);
	tmpBuffer.Store<Data>(offsetY, x);

	Data y = tmpBuffer.Load<Data>(offsetX);
	return y.a*256*16 + int(y.b);
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	outputBuffer[tid] = test(tid);
}
