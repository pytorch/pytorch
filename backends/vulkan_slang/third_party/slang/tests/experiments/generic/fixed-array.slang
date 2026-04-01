//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test around use of an array like container. 

Has error:

.slang(5): error 30019: expected an expression of type 'int', got 'typeof(SIZE)'
    T elements[SIZE];

Problem here is SIZE doesn't have *let* in front. The error doesn't provide any useful insight into that. It's also not entirely clear *why* let is needed. Presumably because it's not a type?
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct FixedArray<T, SIZE : int> 
{   
    T elements[SIZE];
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    FixedArray<int, 8> arr;
    
	outputBuffer[index] = 1;
}

