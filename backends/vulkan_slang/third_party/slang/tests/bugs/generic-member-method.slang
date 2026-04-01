// Test calling a generic member method with partially specified generic arguments.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj
//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface INothing
{
    int getVal();
}

struct Nothing : INothing
{
    int getVal() { return 0; }
}

interface IImpl
{
    int computeAdd<TNothing : INothing, TNothing2 : INothing>(TNothing nothingArg, TNothing2 nothing, int otherVal);
}

struct Impl : IImpl
{
    int thisVal;
    __init()
    {
        thisVal = 1;
    }
    // Introduce two generic parameters so we can construct a partially specialized invoke.
    int computeAdd<TNothing : INothing, TNothing2 : INothing>(TNothing nothingArg, TNothing2 nothing, int otherVal)
    {
        return thisVal + otherVal + nothingArg.getVal();
    }
    int doThing(int otherVal)
    {
        Nothing nothing;
        // Partially specialize the invoke.
        // This will result in construction of `PartiallySpecializedInvokeExpr`.
        // We need to make sure the baseExpr is persisted during this step.
        return computeAdd<Nothing>(nothing, nothing, otherVal);
    }
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    Impl impl;
    Nothing nothing;
    outputBuffer[0] = impl.doThing(2);
}
