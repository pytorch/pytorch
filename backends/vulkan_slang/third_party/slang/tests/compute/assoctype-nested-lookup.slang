

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IFoo : IDefaultInitializable
{
    associatedtype Bar : IFoo;
};


struct FooPair<T : IFoo> : IFoo, IDefaultInitializable
{
    T a;
    T.Bar b;

    typealias Bar = FooPair<T.Bar>;
};


struct ConcreteFoo : IFoo
{
    typealias Bar = ConcreteFoo;

    float x;
};

void test(FooPair<ConcreteFoo>.Bar pair)
{
    pair.a.x = 1.0;
    pair.b.x = 2.0;

    outputBuffer[0] = pair.a.x + pair.b.x;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    FooPair<ConcreteFoo>.Bar pair;
    test(pair);
}
