//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

interface IFoo
{
    This foo();
}

struct Foo : IFoo
{
    float x;
    This foo()
    {
        This f;
        f.x = 0.0;
        return f;
    }
}

RWStructuredBuffer<float> output;

// CHECK: ([[# @LINE+1]]): error 30401
float compute<T:Foo>(T f) // error, should be IFoo here.
{
}

void main()
{}
