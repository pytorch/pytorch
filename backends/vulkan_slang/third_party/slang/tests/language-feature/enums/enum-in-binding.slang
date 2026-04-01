//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpDecorate %g_buffer Binding 2

enum class E {
    A,
    B,
    C
}

[vk::binding(E.C)]
RWStructuredBuffer<float> g_buffer;

[numthreads(1,1,1)]
void main()
{
    g_buffer[0] = 1.0f;
}