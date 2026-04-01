//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test around extension to a built in, to allow associating types with builtins. 

Doesn't work

.slang(22): internal error 99999: unexpected condition encountered in Slang compiler: should not appear in input syntax
    int.Unsigned v = (int.Unsigned)10;

*/


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IIntegral
{
    associatedtype Unsigned;
};

extension int : IIntegral
{
    typedef uint Unsigned;
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    // Works
    int.Unsigned u = 10;

    // Doesn't work
    int.Unsigned v = (int.Unsigned)10;

    outputBuffer[dispatchThreadID.x] = doThing<2>(index);
}