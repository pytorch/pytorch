//TEST:SIMPLE(filecheck=CHECK): -target hlsl -line-directive-mode none -entry computeMain -stage compute

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

struct Foo2 : IDifferentiable
{
    float dx[3];
}

struct Foo : IDifferentiable
{
    typealias Differential = Foo2;

    float x[3];
    // CHECK: tests/autodiff/struct-without-diff-associations.slang(15): error 30102: differentiable member 'x' should have a corresponding field in 'Foo2'. Use [DerivativeMember(Foo2.<field-name>)] or mark as no_diff
    // CHECK-NEXT: float x[3];
    // CHECK-NEXT:       ^
};

[Differentiable]
Foo getFoo(float x)
{
    return { { x, x, x } };
}

[Differentiable]
float foobar(float x)
{
    int i = 3 * int(floor(x));
    Foo foo = getFoo(x);
    return foo.x[i] * foo.x[i];
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        float a = 0.5;
        var d = fwd_diff(foobar)(diffPair(a, 1.0)).d;
        outputBuffer[0] = d;
    }
}