//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpDecorate %g_Instance AliasedPointer

static int* g_Instance = nullptr;

[vk::push_constant] int* dest;

[numthreads(1)]
void ComputeMain(int tid : SV_DispatchThreadID) {
    // int* g_Instance = nullptr;
    if (tid % 2 == 0) g_Instance = dest;
    if (g_Instance) g_Instance[tid] = 123;
}