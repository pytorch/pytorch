//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef float Real;

__generic<let N : int>
struct myvector
{
    vector<Real, N> val;
 
    [TreatAsDifferentiable]
    __init(vector<Real,N> data)
    {
        val = data;
    }
}

extension myvector<3> : MyLinearArithmeticType
{
    [ForwardDifferentiable]
    static myvector<3> ladd(myvector<3> a, myvector<3> b)
    {
        return myvector<3>(a.val + b.val);
    }

    [ForwardDifferentiable]
    static myvector<3> lmul(myvector<3> a, myvector<3> b)
    {
        return myvector<3>(a.val * b.val);
    }

    [ForwardDifferentiable]
    static myvector<3> lscale(float a, myvector<3> b)
    {
        return myvector<3>(a * b.val);
    }

    [ForwardDifferentiable]
    static float ldot(myvector<3> a, myvector<3> b)
    {
        return dot(a.val, b.val);
    }

    [ForwardDifferentiable]
    __init(vector<Real, 3> a)
    {
        val = a;
    }
};


extension myvector<4> : MyLinearArithmeticType
{
    [ForwardDifferentiable]
    static myvector<4> ladd(myvector<4> a, myvector<4> b)
    {
        return myvector<4>(a.val + b.val);
    }

    [ForwardDifferentiable]
    static myvector<4> lmul(myvector<4> a, myvector<4> b)
    {
        return myvector<4>(a.val * b.val);
    }

    [ForwardDifferentiable]
    static myvector<4> lscale(float a, myvector<4> b)
    {
        return myvector<4>(a * b.val);
    }

    [ForwardDifferentiable]
    static float ldot(myvector<4> a, myvector<4> b)
    {
        return dot(a.val, b.val);
    }

    [ForwardDifferentiable]
    __init(vector<Real, 4> a)
    {
        val = a;
    }

};

typedef myvector<3> myfloat3;
typedef myvector<4> myfloat4;

typedef DifferentialPair<Real> dpfloat;

[TreatAsDifferentiable]
interface MyLinearArithmeticType
{
    static This ladd(This a, This b);
    static This lmul(This a, This b);
    static This lscale(Real a, This b);
    static Real ldot(This a, This b);
};

extension myfloat3 : IDifferentiable
{
    typedef myfloat3 Differential;

    [DerivativeMember(Differential.val)]
    extern vector<Real, 3> val;

    static Differential dzero()
    {
        return myfloat3(0);
    }

    [ForwardDifferentiable]
    static Differential dadd(Differential a, Differential b)
    {
        return a + b;
    }

    [ForwardDifferentiable]
    static Differential dmul<T : __BuiltinRealType>(T a, Differential b)
    {
        return myfloat3(__realCast<Real, T>(a) * b.val);
    }

};

extension myfloat4 : IDifferentiable
{
    typedef myfloat4 Differential;

    [DerivativeMember(Differential.val)]
    extern vector<Real, 4> val;

    static Differential dzero()
    {
        return myfloat4(0);
    }

    [ForwardDifferentiable]
    static Differential dadd(Differential a, Differential b)
    {
        return a + b;
    }

    [ForwardDifferentiable]
    static Differential dmul<T: __BuiltinRealType>(T a, Differential b)
    {
        return myfloat4(__realCast<Real, T>(a) * b.val);
    }
};

typedef DifferentialPair<myfloat4> dpfloat4;
typedef DifferentialPair<myfloat3> dpfloat3;

extension float : MyLinearArithmeticType
{
    [ForwardDifferentiable]
    static float ladd(float a, float b)
    {
        return a + b;
    }

    [ForwardDifferentiable]
    static float lmul(float a, float b)
    {
        return a * b;
    }

    [ForwardDifferentiable]
    static float lscale(float a, float b)
    {
        return a * b;
    }
    
    [ForwardDifferentiable]
    static float ldot(float a, float b)
    {
        return a * b;
    }
};

typealias MyLinearArithmeticDifferentiableType = IDifferentiable & MyLinearArithmeticType;

__generic<T : MyLinearArithmeticDifferentiableType>
[ForwardDifferentiable]
T operator +(T a, T b)
{
    return T.ladd(a, b);
}

__generic<T : MyLinearArithmeticDifferentiableType>
[ForwardDifferentiable]
T operator *(T a, T b)
{
    return T.lmul(a, b);
}

__generic<G : MyLinearArithmeticDifferentiableType>
[ForwardDifferentiable]
G f(G x)
{
    G a = x + x;
    G b = x * x;

    return a * a + G.lscale((Real)3.0, x); 
}


[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    {
        dpfloat dpa = dpfloat(2.0, 1.0);
        dpfloat4 dpf4 = dpfloat4(myfloat4(float4(1.5, 2.0, 0.5, 1.0)), myfloat4(float4(0.5, 0.8, 1.6, 2.5)));
        dpfloat3 dpf3 = dpfloat3(myfloat3(float3(1.0, 3.0, 5.0)), myfloat3(float3(0.5, 1.5, 2.5)));

        outputBuffer[0] = f(dpa.p);                       // Expect: 22.0
        outputBuffer[1] = __fwd_diff(f)(dpfloat(2.0, 0.5)).d;  // Expect: 9.5
        outputBuffer[2] = __fwd_diff(f)(dpf4).d.val.w;         // Expect: 27.5
        outputBuffer[3] = __fwd_diff(f)(dpf3).d.val.y;         // Expect: 40.5
    }
}
