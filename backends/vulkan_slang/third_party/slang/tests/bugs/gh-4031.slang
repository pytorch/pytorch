//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -compute -entry computeMain -allow-glsl -profile cs_6_6 -use-dxil -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -output-using-type -emit-spirv-directly

//TEST_INPUT: ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

bool Test_unpackUnorm4x8()
{
    vector<float,4> val;
    val[0] = 1.f / 1.f;
    val[1] = 1.f / 2.f;
    val[2] = 1.f / 4.f;
    val[3] = 1.f / 8.f;

    const uint32_t packed = packUnorm4x8(val);
    const vector<float,4> unpacked = unpackUnorm4x8(packed);

    return true
        && int(unpacked[0] * 1.f) == 1
        && int(unpacked[1] * 2.f) == 1
        && int(unpacked[2] * 4.f) == 1
        && int(unpacked[3] * 8.f) == 1
        ;
}

bool Test_unpackSnorm4x8()
{
    vector<float,4> val;
    val[0] = 1.f / 1.f;
    val[1] = 1.f / 2.f;
    val[2] = 1.f / 4.f;
    val[3] = 1.f / 8.f;

    const uint32_t packed = packSnorm4x8(val);
    const vector<float,4> unpacked = unpackSnorm4x8(packed);

    return true
        && int(unpacked[0] * 1.f) == 1
        && int(unpacked[1] * 2.f) == 1
        && int(unpacked[2] * 4.f) == 1
        && int(unpacked[3] * 8.f) == 1
        ;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    //BUF:1
    outputBuffer[0] = int(true
        && Test_unpackUnorm4x8()
        && Test_unpackSnorm4x8()
        );
}
