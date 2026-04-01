//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target glsl
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -shaderobj -vk

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK-NOT: {{\b}}if{{\b}}

    int tid = dispatchThreadID.x;
    
    static int a[] = { 1, 2 };
    
	outputBuffer[dispatchThreadID.x] = a[tid & 1];
    // BUF: 1
    // BUF: 2
    // BUF: 1
    // BUF: 2
}
