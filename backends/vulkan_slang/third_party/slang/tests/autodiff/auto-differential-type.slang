// Tests automatic synthesis of Differential type requirement.

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct A : IDifferentiable
{
    float x;
    float y;

    [__unsafeForceInlineEarly]
    static Differential dzero()
    {
        Differential b = {0.0, float.dzero()};
        return b;
    }

    [__unsafeForceInlineEarly]
    static Differential dadd(Differential a, Differential b)
    {
        Differential o = {a.x + b.x, 0.0};
        return o;
    }

    [__unsafeForceInlineEarly]
    __generic<T : __BuiltinRealType>
    static Differential dmul(T a, Differential b)
    {
        Differential o = { __realCast<float, T>(a * __realCast<T, float>(b.x)), 0.0};
        return o;
    }
};

typedef DifferentialPair<A> dpA;

[ForwardDifferentiable]
A f(A a)
{
    A aout;
    aout.y = 2 * a.x;
    aout.x = 5 * a.x;

    return aout;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        A a = {1.0, 2.0};
        A.Differential b = {0.2};

        dpA dpa = dpA(a, b);

        outputBuffer[0] = __fwd_diff(f)(dpa).d.x;                    // Expect: 1
    }
}
