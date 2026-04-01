//TEST:SIMPLE(filecheck=CHECK): -target spirv
import "generic-disambiguate-module";

// CHECK: OpEntryPoint

[numthreads(1,1,1)]
void main()
{
    var x : Foo<Foo<Foo<Foo<float>>>>; //   error 20002: syntax error.
}