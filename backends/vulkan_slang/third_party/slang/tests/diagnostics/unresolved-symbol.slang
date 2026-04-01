//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -profile cs_6_0
interface IFoo
{
    int doThing();
}

// CHECK: ([[#@LINE+1]]): error 45001:
extern struct Foo : IFoo;
// CHECK: ([[#@LINE+1]]): error 45001:
extern static const int c;

RWStructuredBuffer<int> output;

void main()
{
    Foo f;
    output[0] = c + f.doThing();
}