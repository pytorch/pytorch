//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

interface Base
{
    static int getValue();
}

struct Impl : Base
{
    // This is not static and not allowed to implement the interface's static method.
    // CHECK: error 38105: member 'getValue' does not match interface requirement
    int getValue() { return 5; }
}

int callGet<Foo : Base>()
{
    return Foo::getValue();
}

RWStructuredBuffer<int> result;

[numthreads(1, 1, 1)]
void computeMain()
{
    result[0] = callGet<Impl>();
}

