//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -xslang -fvk-use-dx-layout -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry computeMain -stage compute -fvk-use-dx-layout

struct Test_t
{
//SPIRV: Offset 0
    uint v0;
//SPIRV: Offset 4
    float3 v1;
//SPIRV: Offset 16
    uint3 v2;
//SPIRV: Offset 28
    uint2 v3;
//SPIRV: Offset 36
    uint2 v4;
//SPIRV: Offset 44
    uint v5[4];
//SPIRV: Offset 60
    uint3 v6[2];
};

//TEST_INPUT:ubuffer(data=[1   2.0 3.0 4.0   5 6 7   8 9 10 11   12 13 14 15   16 17 18   19 20 21], stride=4):out,name testBuffer
RWStructuredBuffer<Test_t> testBuffer;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

__generic<T : IArithmetic, let N : int>
bool comp(vector<T,N> v1, vector<T,N> v2)
{
    for (uint i = 0; i < N; i++)
        if (v1[i] != v2[i])
            return false;

    return true;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeMain()
{
    // CHECK: 64
    Test_t test = testBuffer[0];
    outputBuffer[0] = (true
            && test.v0 == 1
            && comp(test.v1, float3( 2, 3, 4))
            && comp(test.v2, uint3( 5, 6, 7))
            && comp(test.v3, uint2( 8, 9 ))
            && comp(test.v4, uint2( 10, 11 ))
            && test.v5[0] == 12
            && test.v5[1] == 13
            && test.v5[2] == 14
            && test.v5[3] == 15
            && comp(test.v6[0], uint3( 16, 17, 18 ))
            && comp(test.v6[1], uint3( 19, 20, 21 ))
        ) ? 100 : 0;
}