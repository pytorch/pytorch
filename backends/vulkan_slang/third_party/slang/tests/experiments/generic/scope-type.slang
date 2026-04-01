//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test here is to try and associate a value with a type

This may be a similar problem as with scope-value-2.slang. 
This doesn't work *only* if Thing derives from IIntegral. Remove that and it's fine.

.slang(19): internal error 99999: unexpected condition encountered in Slang compiler: should not appear in input syntax
    let v = (Thing::Unsigned)10;
                  ^~

*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IIntegral
{
    associatedtype Unsigned;
};

struct Thing : IIntegral
{
    typedef uint Unsigned;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    // Doesn't work
    let v = (Thing::Unsigned)10;

    outputBuffer[dispatchThreadID.x] = int(v);
}