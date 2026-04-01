// modern-syntax.slang
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -slang -compute -shaderobj

// This file exists to confirm that declarations using "modern"
// syntax are handled correctly by the compiler front-end.

typealias MyInt = int;

func test(val: MyInt) -> MyInt
{
	var tmp = val;
	let c : MyInt = 16;
	tmp += c;
	return tmp;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	int val = test(tid);
	outputBuffer[tid] = val;
}
