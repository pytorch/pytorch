// this-in-extension.slang

// Test that an `This` type works correctly when there is an extension.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -vk

interface IFoo
{
    static const This identity;
}

__generic<T:IFoo>
extension T
{
    This getIdentity()
    {
        return identity;
    }
}

struct FooImpl : IFoo
{
    int v = 1;
    static const This identity = This();   
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    FooImpl foo = {};
    var ident = foo.getIdentity();
    // CHECK: 1
    outputBuffer[0] = ident.v;
}
