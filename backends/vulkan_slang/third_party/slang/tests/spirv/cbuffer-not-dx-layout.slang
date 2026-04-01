//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry computeMain -stage compute

cbuffer Test
{
//SPIRV: Offset 0
    uint v0;

//SPIRV: Offset 16
    float3 v1;

//SPIRV: Offset 32
    uint3 v2;
    
//SPIRV: Offset 48
    uint2 v3;

//SPIRV: Offset 56
    uint2 v4;

//SPIRV: Offset 64
    uint v5[4];

//SPIRV: Offset 128
    uint3 v6[2];
};

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
    outputBuffer[0] = (true
            && v0 == 1
            && comp(v1, float3( 2, 3, 4))
            && comp(v2, uint3( 5, 6, 7))
            && comp(v3, uint2( 8, 9 ))
            && comp(v4, uint2( 10, 11 ))
            && v5[0] == 12
            && v5[1] == 13
            && v5[2] == 14
            && v5[3] == 15
            && comp(v6[0], uint3( 16, 17, 18 ))
            && comp(v6[1], uint3( 19, 20, 21 ))
        ) ? 100 : 0;
}