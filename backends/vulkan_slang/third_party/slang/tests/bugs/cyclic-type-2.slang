
//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly

// CHECK: error 41001:

interface IFoo {}
struct TA<T:IFoo> : IFoo
{
    TB<T> tb;
}

struct TB<T:IFoo>
{
    TA<T> ta[2];
}

struct TC : IFoo {}

[numthreads(1,1,1)]
void main()
{
    TA<TC> ta;
    ta.tb.ta[0] = ta;
}