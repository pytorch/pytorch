//TEST:SIMPLE(filecheck=CHECK):-target spirv

// CHECK: OpEntryPoint

struct Foo{ float4 v; }

[shader("vertex")]
float4 vert(in Foo in, in Foo o) : SV_Position
{
    return in.v;
}