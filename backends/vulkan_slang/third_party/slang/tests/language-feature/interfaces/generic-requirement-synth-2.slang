// Test that we allow type conformances whose base interface is generic.

//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-dx11 -compute  -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute  -output-using-type

interface IStack<let D : int>
{
    IStack<D - N> popN<let N : int>();

    int get();
}
struct StackImpl<let D : int> : IStack<D>
{
    // member 'popN' does not match interface requirement.
    StackImpl<D - N> popN<int N>() { return StackImpl<D - N>(); }

    int get() { return D; }
}

int helper<int n, T : IStack<n>>(T stack)
{
    return stack.popN<2>().get();
}

//TEST_INPUT: set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4);
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain()
{
    StackImpl<5> obj = StackImpl<5>();

    // CHECK: 3
    outputBuffer[0] = helper(obj);
}