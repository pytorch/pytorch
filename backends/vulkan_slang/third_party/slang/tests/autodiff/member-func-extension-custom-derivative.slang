//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A
{
    float x;

    float f(float v)
    {
        return v * v;
    }
}

extension A
{
    [ForwardDerivativeOf(f)]
    DifferentialPair<float> diff_f(DifferentialPair<float> v)
    {
        return diffPair(v.p * v.p, v.p * v.d * 2.0);
    }
}

struct Foo<T : IDifferentiable>
{
    T value;
    T doThing() { return value; }
}

extension<T : IDifferentiable> Foo<T>
{
    [ForwardDerivativeOf(doThing)]
    DifferentialPair<T> diff_doThing()
    {
        return diffPair(value, T.dzero());
    }
}


[ForwardDifferentiable]
float test(Foo<float> obj, float v)
{
    return obj.doThing() * v;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Foo<float> a = {0.0};
    var p = diffPair(3.0, 1.0);
    let rs = __fwd_diff(test)(a, p);
    outputBuffer[0] = rs.d;
}
