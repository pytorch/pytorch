
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUFFER):-vk -compute -output-using-type
//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry computeMain -emit-spirv-directly

//TEST_INPUT:set output = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<float> output;

// Test that we are able to use the VectorTimesScalar opcode to simplify the resulting spirv.

// CHECK: OpVectorTimesScalar

[numthreads(1,1,1)]
void computeMain(int3 tid : SV_DispatchThreadID)
{
    float3 v = tid + 2.0;
    float3 v1 = v * 0.5;
    // BUFFER: 1.0
    output[0] = v1.x;
}