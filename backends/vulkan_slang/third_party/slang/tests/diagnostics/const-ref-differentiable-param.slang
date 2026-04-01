//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):


[Differentiable]
float f(__constref float3 val)
{
    return val.x;
}

struct MyType : IDifferentiable
{
    // Error: cannot use constref on a differentiable member method of a differentiable type.
    [Differentiable]
    [constref] float compute(float x) { return 0; }

    // OK
    [Differentiable]
    float compute1(float x) { return 0; }

    // OK
    [constref] float compute2(float x) { return 0;}
}

struct MyType2
{
    // OK.
    [Differentiable]
    [constref] float compute(float x) { return 0; }

    // OK
    [constref]
    float compute1(float x) { return 0; }
}

// CHECK-DAG: {{.*}}(5): error 38034: cannot use '__constref' on a differentiable parameter.
// CHECK-NOT {{.*}}error
// CHECK-DAG: {{.*}}(14): error 38034: cannot use '[constref]' on a differentiable member method of a differentiable type.
// CHECK-NOT {{.*}}error
