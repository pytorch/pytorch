//TEST(smoke,compute):COMPARE_COMPUTE:-cpu -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct A<let I : int>
{
    int f() { return I; }
};

struct B<let U : uint> : IDefaultInitializable
{
    A<U> a;
};

int foo<let I : int>(A<I> a)
{
    return a.f();
}

int bar<let U : uint>(B<U> b)
{
    return foo(b.a);
    // We previously were inferring the type at which to call `foo` as `U`
    // instead of `int(U)`. This then cause the typechecker to impmediately
    // fail because `U` does not unify with the type of `B<U>`'s `a` member,
    // namely `int(U)`.
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    A<1> a;
    B<1> b;
    b.a = a;
    outputBuffer[dispatchThreadID.x] = bar<1>(b);
}
