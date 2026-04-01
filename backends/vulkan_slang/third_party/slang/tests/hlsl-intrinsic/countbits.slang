//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -cpu
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -dx11
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -dx12
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -vk -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -cuda -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -mtl -compute
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -wgpu -compute
// TODO: test GLSL pathway once emit-spirv-via-glsl is fixed and shader output reading is fixed for GLSL

//CHK:1

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint r1 = countbits(0b1);
    uint2 r2 = countbits(uint2(0b0, 0b1));
    uint3 r3 = countbits(uint3(0b0, 0b1, 0b11));
    uint4 r4 = countbits(uint4(0b0, 0b1, 0b11, 0b111));

    outputBuffer[0] = true
        && (r1 == 1)
        && (r2.x == 0 && r2.y == 1)
        && (r3.x == 0 && r3.y == 1 && r3.z == 2)
        && (r4.x == 0 && r4.y == 1 && r4.z == 2 && r4.w == 3)
        ;
}
