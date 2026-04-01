//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// CHECK: cyclic reference
interface IFoo { void ff(); }
interface I : IFoo,K { void doThing(); } // ': I' is the problem.
interface J : K {}
interface K : I { void doThingK(); }
struct S<T: I> {
 T t;
}

void t<T : I>(T j)
{
    j.doThing();
}

[shader("compute")]
[numthreads(1,1,1)]
void main()
{

}