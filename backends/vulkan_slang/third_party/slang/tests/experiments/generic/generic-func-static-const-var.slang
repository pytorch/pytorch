//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -Xslang -dump-ir

/* Doesn't work correctly failing in emit with a 'generic' instruction that is not expected. */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;
    
struct GetValue2<let N : int>
{
    static int getValue()
    {
        static const int v = N;
        return v;
    }
};    
  
[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

	outputBuffer[index] = GetValue2<20>::getValue(); 
}

