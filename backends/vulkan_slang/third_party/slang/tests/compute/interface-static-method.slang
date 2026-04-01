// interface-static-method.slang

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj

interface IHideout
{
	int getAnimalCount();	
}

interface ISuperhero
{
	associatedtype Hideout : IHideout;

	static Hideout getHideout();
}

struct Batcave : IHideout
{
	int batCount;	

	int getAnimalCount() { return batCount; }
}

struct Batman : ISuperhero
{
	typedef Batcave Hideout;

	static Batcave getHideout()
	{
		Batcave batcave = { 100 };
		return batcave;
	}
}

int doIt<T:ISuperhero>()
{
	return T.getHideout().getAnimalCount();
}

int test(int val)
{
	return doIt<Batman>();
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	outputBuffer[tid] = test(tid);
}
