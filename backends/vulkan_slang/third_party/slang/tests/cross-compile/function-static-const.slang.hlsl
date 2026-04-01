// function-static-const.slang
//TEST_IGNORE_FILE

#pragma pack_matrix(column_major)

struct SLANG_ParameterGroup_C_0
{
    int index_0;
};

cbuffer C_0 : register(b0)
{
    SLANG_ParameterGroup_C_0 C_0;
}

static const int kArray_0[int(16)] = { int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9), int(10), int(11), int(12), int(13), int(14), int(15), int(16) };

int test_0(int val_0)
{
    return kArray_0[val_0];
}

float4 main() : SV_TARGET
{
    int _S1 = test_0(C_0.index_0);
    return (float4) float(_S1);
}
