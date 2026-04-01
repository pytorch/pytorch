//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test using interfaces with mutation.

Doesn't work because

.slang(18): error 30050: mutating method 'lock' cannot be called on an immutable value
    lockable.lock();
    
If that section is removed we get on `doThing(count)`

.slang(31): error 30047: argument passed to parameter '0' must be l-value.

Which perhaps means an inteface cannot be used as a lvalue?
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface ILockable
{
    [mutating] void lock();
    [mutating] void unlock();
};

struct IntCount : ILockable
{
    [mutating] void lock() { count ++; }
    [mutating] void unlock() { --count; }
    int count = 0;
};

int doThing(inout ILockable lockable)
{
    lockable.lock();
    // do something...
    
    lockable.unlock();
    return 0;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    IntCount count;
    doThing(count);
    
    outputBuffer[dispatchThreadID.x] = count.count;
}