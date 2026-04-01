// groupshared.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out, name=gBuffer
RWStructuredBuffer<int> gBuffer;

static const int A = 4;
static const int B = 1;
static const int THREAD_COUNT = A * B;

groupshared int gA[THREAD_COUNT];
int test(int val)
{
	gA[val] = val;
	GroupMemoryBarrierWithGroupSync();
	val = gA[val ^ 1];

/* TODO: once function-scope `static` works
	static groupshared int gB[THREAD_COUNT];

	gB[val] = val;
	GroupMemoryBarrierWithGroupSync();
	val = gB[val ^ 2];
*/

	return val;
}

[numthreads(THREAD_COUNT, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	int val = int(tid);
	val = test(val);
	gBuffer[tid] = val;
}