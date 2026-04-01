//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -xslang -fvk-use-dx-layout -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry computeMain -stage compute -fvk-use-dx-layout 
//TEST_INPUT:cbuffer(data=[1 0 0 0   2.0 3.0 4.0 0   5.0 6.0 7.0 0   8.0 9.0 10.0 11]):name=Test

//SPIRV: ArrayStride 16

cbuffer Test
{
//SPIRV: Offset 0
    uint v0;

//SPIRV: Offset 16
// matrix always start on a new register
    float3x3 v1;
//SPIRV: Offset 60
// Non-matrix can pack with a partially filled register
    uint v2;
};

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
[numthreads(2, 2, 1)]
void computeMain()
{
    // CHECK: 64

    outputBuffer[0] = (true
            && v0 == 1
            && comp(v1[0], float3(2, 3, 4))
            && comp(v1[1], float3(5, 6, 7))
            && comp(v1[2], float3(8, 9, 10))
            && v2 == 11
        ) ? 100 : 0;
}