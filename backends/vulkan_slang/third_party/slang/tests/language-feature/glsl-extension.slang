//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry main -stage compute

RWStructuredBuffer<float> outputBuffer;

// Check that extensions on types work
__glsl_extension(GL_BAR_bar)
// CHECK-DAG: #extension GL_BAR_bar : require
struct S
{
    float a;
}

// Check that extensions on functions work
__glsl_extension(GL_FOO_foo)
// CHECK-DAG: #extension GL_FOO_foo : require
float foo(S s)
{
    return s.a;
}

[numthreads(4, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    S s;
    s.a = 0;
    outputBuffer[dispatchThreadID.x] = foo(s);
}
