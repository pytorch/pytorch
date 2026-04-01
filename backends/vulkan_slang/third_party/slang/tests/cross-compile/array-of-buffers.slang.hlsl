#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif
#pragma warning(disable: 3557)

struct SLANG_ParameterGroup_C_0
{
    uint index_0;
};

cbuffer C_0 : register(b0)
{
    SLANG_ParameterGroup_C_0 C_0;
}
struct S_0
{
    float4 f_0;
};

ConstantBuffer<S_0 >  cb_0[int(3)] : register(b1);

StructuredBuffer<S_0 >  sb1_0[int(4)] : register(t0);

RWStructuredBuffer<float4 >  sb2_0[int(5)] : register(u0);

ByteAddressBuffer  bb_0[int(6)] : register(t4);
float4 main() : SV_TARGET
{

    float4 _S1 = cb_0[C_0.index_0].f_0 + sb1_0[C_0.index_0].Load(C_0.index_0).f_0 + sb2_0[C_0.index_0][C_0.index_0];
    uint _S2 = bb_0[C_0.index_0].Load(int(C_0.index_0 * 4U));
    return _S1 + (float4)float(_S2);
}

