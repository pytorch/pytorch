//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d12 -shaderobj -output-using-type -use-dxil
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type

//TEST_INPUT:set outputBuffer = out ubuffer(data=[0 0 0 0], stride=4)
RWStructuredBuffer<uint> outputBuffer;

interface ITestB
{
    static void eval<let N : uint>(uint val[N], out uint res[N]);
};

struct TestB : ITestB
{
    static void eval<let N : uint>(uint val[N], out uint res[N])
    {
        static const uint scale = 3 * N;
        for (uint i = 0; i < N; i++)
            res[i] = scale * val[i];
    }
};

[numthreads(1, 1, 1)]
void computeMain(uint3 threadID: SV_DispatchThreadID)
{
    const uint i = threadID.x;

    uint val[4] = { i, i + 1, i + 2, i + 3 };
    uint res[4];

    TestB test;
    test.eval<4>(val, res);

    // CHECK: 0
    outputBuffer[0] = res[0];
    // CHECK: 12
    outputBuffer[1] = res[1];
    // CHECK: 24
    outputBuffer[2] = res[2];
    // CHECK: 36
    outputBuffer[3] = res[3];
}