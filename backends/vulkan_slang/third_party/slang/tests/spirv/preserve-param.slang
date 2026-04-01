//TEST:SIMPLE(filecheck=CHECK): -target spirv -preserve-params -O0

// Test that -preserve-params option preserves shader paraemter declarations in the output spirv code.

// CHECK-DAG: OpDecorate %{{.*}} Binding 0

// CHECK-DAG: OpDecorate %{{.*}} Binding 1

RWStructuredBuffer<float> buffer;

struct TT
{
    Texture2D tex;    
}
[numthreads(1, 1, 1)]
void f(ConstantBuffer<TT> t)
{
    return;
}