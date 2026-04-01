//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

typedef float Real;

typealias IDFloat = __BuiltinRealType & IDifferentiable;

__generic<T : IDifferentiable, let N : int>
struct dvector : IDifferentiable
{
    typedef dvector<T.Differential, N> Differential;
    [DerivativeMember(Differential.values)]
    T values[N];
};

__generic<T : IDFloat, let N : int>
struct myvector : IDifferentiable
{
    typedef dvector<T.Differential, N> Differential;

    [DerivativeMember(Differential.values)]
    T values[N];
    
    __init(T c)
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            values[i] = c;
        }
    }

    static Differential dadd(Differential a, Differential b)
    {
        Differential output;

        for (int i = 0; i < N; i++)
        {
            output.values[i] = T.dadd(a.values[i], b.values[i]);
        }

        return output;
    }

    
    static Differential dmul<U: __BuiltinRealType>(U a, Differential b)
    {
        Differential output;

        for (int i = 0; i < N; i++)
        {
            output.values[i] = T.dmul<U>(a, b.values[i]);
        }

        return output;
    }

    static Differential dzero()
    {
        Differential output;

        for (int i = 0; i < N; i++)
        {
            output.values[i] = T.dzero();
        }

        return output;
    }
};

[ForwardDifferentiable]
__generic<T : IDFloat, let N : int>
myvector<T, N> operator +(myvector<T, N> a, myvector<T, N> b)
{
    myvector<T, N> output;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        output.values[i] = a.values[i] + b.values[i];
    }
    return output;
}

[ForwardDifferentiable]
__generic<T : IDFloat, let N : int>
myvector<T, N> operator *(myvector<T, N> a, myvector<T, N> b)
{
    myvector<T, N> output;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        output.values[i] = a.values[i] * b.values[i];
    }
    return output;
}

[ForwardDifferentiable]
__generic<T : IDFloat, let N : int>
myvector<T, N> operator *(T a, myvector<T, N> b)
{
    myvector<T, N> output;
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        output.values[i] = a * b.values[i];
    }
    return output;
}

__generic<T : IDFloat, let N : int>
[ForwardDerivative(dot_jvp)]
T dot(myvector<T, N> a, myvector<T, N> b)
{
    T curr = __realCast<T, float>(0.f);
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        curr = curr + (a.values[i] * b.values[i]);
    }

    return curr;
}

__generic<T : IDFloat, let N : int>
typedef DifferentialPair<myvector<T, N>> dpvector;

__generic<T : IDFloat, let N : int>
DifferentialPair<T> dot_jvp(dpvector<T, N> a, dpvector<T, N> b)
{
    T.Differential curr_d = (T.dzero());
    T curr_p = __realCast<T, float>(0.f);
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        curr_p = curr_p + (a.p.values[i] * b.p.values[i]);
        curr_d = T.dadd(
                    curr_d, 
                    T.dadd(
                        T.dmul<T>(a.p.values[i], b.d.values[i]),
                        T.dmul<T>(b.p.values[i], a.d.values[i])));
    }

    return DifferentialPair<T>(curr_p, curr_d);
}

__generic<let N : int>
struct lineardvector : IDifferentiable
{
    typedef lineardvector<N> Differential;

    myvector<Real, N>.Differential val;

    __init(vector<Real.Differential, N> a)
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            val.values[i] = a[i];
        }
    }

    // Add a new constructor for dadd() function.
    __init(Real a[N])
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            val.values[i] = a[i];
        }
    }
};

__generic<let N : int>
struct linearvector : MyLinearArithmeticType, IDifferentiable
{
    typedef lineardvector<N> Differential;

    [DerivativeMember(Differential.val)]
    myvector<Real, N> val;

    [ForwardDifferentiable]
    static linearvector<N> ladd(linearvector<N> a, linearvector<N> b)
    {
        return linearvector<N>(a.val + b.val);
    }

    [ForwardDifferentiable]
    static linearvector<N> lmul(linearvector<N> a, linearvector<N> b)
    {
        return linearvector<N>(a.val * b.val);
    }

    [ForwardDifferentiable]
    static linearvector<N> lscale(float a, linearvector<N> b)
    {
        return linearvector<N>(a * b.val);
    }

    [ForwardDifferentiable]
    static float ldot(linearvector<N> a, linearvector<N> b)
    {
        return dot(a.val, b.val);
    }

    static Differential dzero()
    {
        lineardvector<N> dout;
        dout.val = myvector<Real, N>.dzero();
        return dout;
    }

    static Differential dadd(Differential a, Differential b)
    {
        // return { myvector<Real, N>.dadd(a.val, b.val) };
        //
        // Above code will not work because
        // myvector<Real, N>.dadd will return dvector<T.Differential, N> type
        // while Differential == lineardvector<N> type
        // and the constructor of lineardvector<N> requires a vector<Real.Differential, N> type
        // and dvector<T.Differential, N> != vector<Real.Differential, N>, though they have the
        // same members.
        //
        // In our new design, generic will not be C-Style struct anymore.
        dvector<Real.Differential, N> d = myvector<Real, N>.dadd(a.val, b.val);
        return {d.values};
    }

    static Differential dmul<T: __BuiltinRealType>(T a, Differential b)
    {
        dvector<Real.Differential, N> d = myvector<Real, N>.dmul<T>(a, b.val);
        return {d.values};
    }

    [ForwardDifferentiable]
    __init(vector<Real, N> a)
    {
        [ForceUnroll]
        for (int i = 0; i < N; i++)
        {
            val.values[i] = a[i];
        }
    }

    [ForwardDifferentiable]
    __init(myvector<Real, N> a)
    {
        val = a;
    }
};

typedef linearvector<3> myfloat3;
typedef linearvector<4> myfloat4;

typedef lineardvector<3> mydfloat3;
typedef lineardvector<4> mydfloat4;

typedef DifferentialPair<Real> dpfloat;

[TreatAsDifferentiable]
interface MyLinearArithmeticType
{
    static This ladd(This a, This b);
    static This lmul(This a, This b);
    static This lscale(Real a, This b);
    static Real ldot(This a, This b);
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
        dpfloat4 dpf4 = dpfloat4(myfloat4(float4(1.5, 2.0, 0.5, 1.0)), mydfloat4(float4(0.5, 0.8, 1.6, 2.5)));
        dpfloat3 dpf3 = dpfloat3(myfloat3(float3(1.0, 3.0, 5.0)), mydfloat3(float3(0.5, 1.5, 2.5)));

        outputBuffer[0] = f(dpa.p);                               // Expect: 22.0
        outputBuffer[1] = __fwd_diff(f)(dpfloat(2.0, 0.5)).d;          // Expect: 9.5
        outputBuffer[2] = __fwd_diff(f)(dpf4).d.val.values[3];         // Expect: 27.5
        outputBuffer[3] = __fwd_diff(f)(dpf3).d.val.values[1];         // Expect: 40.5
    }
}
