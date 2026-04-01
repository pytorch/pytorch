//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Test use of a generic with a built in doing something simple. 

Doesn't work because cannot find +.

.slang(3): error 39999: no overload for '+' applicable to arguments of type (typeof(V), int)
int doThing<V : int>(int b) { return V + b; }

NOTE! We can fix by adding *let*

int doThing<let V : int>(int b) { return V + b; }
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

int doThing<V : int>(int b) { return V + b; }

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    outputBuffer[dispatchThreadID.x] = doThing<2>(index);
}