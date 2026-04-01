//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -profile glsl_460
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute

// Check that when generating spirv directly, we use a loop
// to copy large arrays in a local variable to a buffer, instead of emitting
// unrolled code that reads each element of the array individually.

struct WorkData
{
    int B[1024];
};

//TEST_INPUT:set resultBuffer = out ubuffer(data=[0 0 0 0], stride=4, count=1024)
RWStructuredBuffer<WorkData> resultBuffer;

// CHECK: OpLoopMerge
// CHECK: OpLoopMerge

// BUF: 0
// BUF: 1
[numthreads(1, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
    WorkData wd;
    for (int i = 0; i < 1024; i++)
        wd.B[i] = i;
    resultBuffer[0] = wd;
}
