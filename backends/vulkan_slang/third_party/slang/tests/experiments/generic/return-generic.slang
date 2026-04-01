//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test of a generic function returning a generic struct.  

Since T could be any type, the compiler can't determine if it's convertable from an int. That may be surprising.

Fails because:

.slang(11): error 30019: expected an expression of type 'T', got 'int'
    a.value = i;
              ^
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Another<T>
{
    T value;
};

Another<T> getAnother<T>(int i)
{
    Another<T> a;
    a.value = i;
    return a;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    Another<int> a = getAnother<int>(index);
    
	outputBuffer[index] = a.value;
}

