// enum-conversion2.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

// More significant stress testing of `enum` types

enum Flower
{
	Rose,
	Violet,
	Tulip,
	Orchid,
	Chrysanthemum,
}

int test(int val)
{
	Flower a = Flower(val);
	Flower b = Flower(uint(val));

	uint x = uint(a);
	int y = int(b);

	uint z = (x + y)/2 + uint(Flower.Violet);

	Flower c = Flower(z);
	return int(c);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    int val = int(tid);
    val = test(val);

    outputBuffer[tid] = val;
}