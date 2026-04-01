//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -profile glsl_460

struct WorkData {
    float A[2048 * 2048];
    float B[2048 * 2048];
};
struct PushData {
    WorkData* Input;
    float* Dest;
};

[vk::push_constant] ConstantBuffer<PushData> cb;

// CHECK: OpEntryPoint

[numthreads(64, 1, 1)]
void ComputeMain(uint tid: SV_DispatchThreadID)
{
    cb.Dest[tid] = cb.Input->A[tid] * cb.Input->B[tid];
}