//DIAGNOSTIC_TEST:SIMPLE:

struct DiffT : IDifferentiable
{
    float f;
}
struct NoDiffField
{
    DiffT fp;
}

[BackwardDifferentiable]
float g(float x)
{
    NoDiffField obj;
    obj.fp.f = x * x; // Error, this location cannot hold derivative.
    return obj.fp.f;
}

[BackwardDifferentiable]
void diffOut(inout float x)
{
    x = 2;
}

float noDiffFunc(float x)
{
    return 0.0;
}

[BackwardDifferentiable]
float h(float x)
{
    NoDiffField obj;
    obj.fp.f = detach(x * x); // OK.
    obj.fp.f = noDiffFunc(x); // OK.
    diffOut(obj.fp.f); // Error.
    return obj.fp.f;
}
