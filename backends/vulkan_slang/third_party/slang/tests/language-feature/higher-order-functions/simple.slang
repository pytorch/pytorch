//TEST(compute, vulkan):COMPARE_COMPUTE_EX():-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX():-slang -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-cpu -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

func twice<A>(f : functype (A) -> A, a : A) -> A
{
    return f(f(a));
}

func apply2<A, B, C>(f : functype (A, B) -> C, a : A, b : B) -> C
{
    return f(a, b);
}

func multiply(x : float, y : float) -> float
{
    return x * y;
}

func square(x : float) -> float
{
    return apply2<float, float, float>(multiply, x, x);
}

[numthreads(4, 1, 1)]
void computeMain(uint tig : SV_GroupIndex)
{
    outputBuffer[tig] = uint(twice(square, -2.f));
}
