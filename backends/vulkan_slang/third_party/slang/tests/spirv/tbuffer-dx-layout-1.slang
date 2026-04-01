//IGNORE_TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry computeMain -stage compute -fvk-use-dx-layout

// 'tbuffer' is not implemented for SPIRV targets currently.

tbuffer Test
{
//SPIRV: Offset 0
    uint v0;
//SPIRV: Offset 4
    float3 v1;

//SPIRV: Offset 16
    uint3 v2;

//SPIRV: Offset 32
    uint2 v3;
//SPIRV: Offset 40
    uint2 v4;

//SPIRV: Offset 48
    uint v5[4];

// array always starts on a new register.
//SPIRV: Offset 112
    uint3 v6[2];
//SPIRV: Offset 140
// non-array can pack with a partially filled register
    uint v7;

//SPIRV: Offset 144
    uint2 v8;

// SPIRV: Offset 160
// array always starts on a new register.
    uint v9[2];
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
[numthreads(2, 2, 1)]
void computeMain()
{
    // CHECK: 64

    outputBuffer[0] = (true
            && v0 == 1
            && comp(v1, float3(2, 3, 4))
            && comp(v2, uint3(5, 6, 7))
            && comp(v3, uint2(8, 9))
            && comp(v4, uint2(10, 11))
            && v5[0] == 12
            && v5[1] == 13
            && v5[2] == 14
            && v5[3] == 15
            && comp(v6[0], uint3(16, 17, 18))
            && comp(v6[1], uint3(19, 20, 21))
            && v7 == 22
            && comp(v8, uint2(23, 24))
            && v9[0] == 25
            && v9[1] == 26
        ) ? 100 : 0;
}