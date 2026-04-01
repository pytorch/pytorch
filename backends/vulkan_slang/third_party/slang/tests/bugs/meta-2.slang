//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct GetValue<let N : int>
{
    static const int Value = N;
};
  
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = int(dispatchThreadID.x);

	outputBuffer[index] = GetValue<10>::Value;
}

