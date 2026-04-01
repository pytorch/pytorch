//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct B : IDifferentiable
{
    typedef B Differential;

    float z;
};

struct A : IDifferentiable
{
    typedef B Differential;
    
    [DerivativeMember(B.z)]
    float x;
    no_diff float y;

    [__unsafeForceInlineEarly]
    static Differential dzero()
    {
        B b = {0.0};
        return b;
    }

    [__unsafeForceInlineEarly]
    static Differential dadd(Differential a, Differential b)
    {
        B o = {a.z + b.z};
        return o;
    }

    [__unsafeForceInlineEarly]
    static Differential dmul<T : __BuiltinRealType>(T a, Differential b)
    {
        B o = {__realCast<float, T>(a) * b.z};
        return o;
    }
};

typedef DifferentialPair<A> dpA;

[ForwardDifferentiable]
A f(A a)
{
    A aout;
    aout.y = detach(2 * a.x);
    aout.x = 5 * a.x;

    return aout;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        A a = {1.0, 2.0};
        B b = {0.2};

        dpA dpa = dpA(a, b);

        outputBuffer[0] = __fwd_diff(f)(dpa).d.z;                    // Expect: 1
    }
}
