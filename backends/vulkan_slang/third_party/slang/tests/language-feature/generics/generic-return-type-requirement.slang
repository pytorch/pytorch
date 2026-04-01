//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type
//TEST:SIMPLE(filecheck=HLSL): -target hlsl -profile cs_6_0 -entry computeMain

// HLSL-NOT: AnyValue

interface IStack<let D : int>
{
    __generic<let N : int>
    IStack<D - N> popN();

    int get();
}
struct StackImpl<let D : int> : IStack<D>
{
    // We should be able to specialize the callsites of this function to use
    // the concrete type instead of resorting to dynamic dispatch.
    __generic<let N : int>
    IStack<D - N> popN() { return StackImpl<D-N>(); }

    int get() { return D; }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

int test<int D, S : IStack<D>>(S stack)
{
    return stack.popN<2>().get();
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    StackImpl<5> stack;

    // CHECK: 2
    outputBuffer[0] = test(stack);
}
