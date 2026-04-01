//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test is to try and see what effects type constraints have. 

This is a bit of a silly example.

Fails with 

.slang(23): note 99999: an internal error threw an exception while working on code near this location
(0): error 99999: Slang compilation aborted due to an exception of class Slang::InternalError: unimplemented: value lowering

*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum XXX {};

__generic<T : XXX>
Enum getEnum(T i)
{
    return Enum::A;
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    XXX x;
    let e = getEnum(x);

    outputBuffer[index] = int(e);
}