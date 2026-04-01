//TEST(compute):COMPARE_COMPUTE: -shaderobj
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj

// Check that user code can declare and use a generic
// `struct` type.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

__generic<T>
struct GenStruct
{
	T x;
	T y;
};

__generic<T>
T test(T val)
{
	GenStruct<T> gs;
	gs.x = val;
	gs.y = val;
	return gs.x;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	float outVal = 0;

	outVal += test<uint>(tid);
	outVal += test<float>(tid * 16.0f);

	outputBuffer[tid] = int(outVal);
}