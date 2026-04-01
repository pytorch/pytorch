//TEST(smoke,compute):COMPARE_COMPUTE: -shaderobj
//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj

// Test overload resolution for nested generic definitions

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;


__generic<T : __BuiltinFloatingPointType>
struct Foo
{
	T test(uint index, T x)
    {
        return __realCast<T, float>(1.f);
	}

	__generic<let N: int>
	T test(vector<uint, N> index, T x)
    {
        return __realCast<T, float>(2.f);
	}
};


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x + 2;
	
	Foo<float> obj;

	float outVal = obj.test(tid, 0.f);
	outputBuffer[0] = outVal; // Expect: 1

	float outVal2 = obj.test(uint2(tid, tid), 0.f);
	outputBuffer[1] = outVal2; // Expect: 2
}