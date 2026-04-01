//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* This test trys to specialize an algorithm through generic parameters. 

Using parameters in this way ensures the algorithm in function form 
will have those values known at compile time, and presumably lead to a specialized version.

Here (with 1 == 1) it's testing if a compile time knowable expression can be used.

Does not work because

.slang(25): note 99999: an internal error threw an exception while working on code near this location
(0): error 99999: Slang compilation aborted due to an exception of class Slang::InternalError: unexpected: ErrorType
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum class Enum
{
    A, B
};

// NOTE! Here we don't have let!
int doThing<a : bool, b : int, c : Enum>()
{
    //if (a)
    {
        return 1;
    }
    return 0;
    //return a ? b + b + int(c) : b - int(c);
}


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    let v = doThing<1 == 1, false, Enum::A>();
    
    outputBuffer[dispatchThreadID.x] = v;
}