// func-cbuffer-param.slang

// Test that passing a `ConstantBuffer<X>` parameter
// into a function works across all target.

// This has a compatibility issue on Windows 10.0.10586 on Dx12 - dxcompiler will crash (can remove form tests with -exclude compatibility-issue)

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan, compatibility-issue):COMPARE_COMPUTE_EX:-dx12 -compute -use-dxil -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj


struct Data
{
	int4 val[4];
}

//TEST_INPUT:cbuffer(data=[0 1 2 3  16 17 18 19  32 33 34 35  48 49 50 51]):name=a
ConstantBuffer<Data> a;

//TEST_INPUT:cbuffer(data=[16 17 18 19  32 33 34 35  48 49 50 51  64 65 66 67]):name=b
ConstantBuffer<Data> b;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

int helper(ConstantBuffer<Data>  buffer, int index)
{
	return buffer.val[index].x;
}

int test(int val)
{
	return val + helper(a, val) + helper(b, val);
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int inVal = (int) dispatchThreadID.x;
    int outVal = test(inVal);
    outputBuffer[dispatchThreadID.x] = outVal;
}