//TEST:SIMPLE(filecheck=SPIRV): -target spirv -emit-spirv-directly

//SPIRV: ArrayStride 12

struct Test
{
//SPIRV: Offset 0
    uint v0;

//SPIRV: Offset 4
// matrix always start on a new register
    float3x3 v1;
//SPIRV: Offset 40
// Non-matrix can pack with a partially filled register
    uint v2;
};

ConstantBuffer<Test, ScalarDataLayout> buffer;

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
            && buffer.v0 == 1
            && comp(buffer.v1[0], float3(2, 3, 4))
            && comp(buffer.v1[1], float3(5, 6, 7))
            && comp(buffer.v1[2], float3(8, 9, 10))
            && buffer.v2 == 11
        ) ? 100 : 0;
}