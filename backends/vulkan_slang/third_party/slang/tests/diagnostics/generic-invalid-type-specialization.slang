//DIAGNOSTIC_TEST:SIMPLE:

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Check<T>
{
    T v;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    // Invalid as should only accept a type
    Check<2 + 2> v; 
    
    outputBuffer[index] = index; 
}
