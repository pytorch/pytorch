//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -output-using-type

interface IValueGeneric<let D : int> {}
struct ValGenericImpl<let D : int> : IValueGeneric<D> {}

struct NestedValueGeneric<let D : int, S : IValueGeneric<D>>
{
    int x;
}

void acceptor<let D : int, S : IValueGeneric<D>>(NestedValueGeneric<D, S> x)
{
    outputBuffer[0] = D + x.x;
}
void test(NestedValueGeneric<2, ValGenericImpl<2>> x)
{
    // Test that we can correctly infer acceptor.D and acceptor.S from `x`.
    acceptor(x);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    NestedValueGeneric<2, ValGenericImpl<2>> x;
    x.x = 1;
    test(x);
    // CHECK: 3
}