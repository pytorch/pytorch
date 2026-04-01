//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct B : IDifferentiable
{
    typedef B Differential;
    float3 z;
    float.Differential k[10];
};

struct A : IDifferentiable
{
    typedef B Differential;

    [DerivativeMember(B.z)]
    float3 x;
    [DerivativeMember(B.k)]
    float y[10];

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
    static Differential dmul<T: __BuiltinRealType>(T a, Differential b)
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

    aout.y[5] = (2 * a.x).y;
    aout.y[2] = (3 * a.y[4]);
    aout.x = float3(5 * a.x.z, 3 * a.x.y, 0.5 * a.x.x);

    return aout;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        float arr[10] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
        A a = {float3(1.0, 2.0, 3.0), arr};

        float d_arr[10] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
        B b = {float3(1.0, 0.5, 0.3), d_arr};

        dpA dpa = dpA(a, b);

        outputBuffer[0] = __fwd_diff(f)(dpa).d.z.z;  // Expect: 0.5
        outputBuffer[1] = __fwd_diff(f)(dpa).d.k[5]; // Expect: 1
        outputBuffer[2] = __fwd_diff(f)(dpa).d.k[2]; // Expect: 1.5
    }
}
