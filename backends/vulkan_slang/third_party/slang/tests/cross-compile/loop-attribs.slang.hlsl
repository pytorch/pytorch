#pragma pack_matrix(column_major)
#ifdef SLANG_HLSL_ENABLE_NVAPI
#include "nvHLSLExtns.h"
#endif
#pragma warning(disable: 3557)

float4 main() : SV_TARGET
{
    float _S1 = 0.0;
    int i_0 = int(0);
    float sum_0 = 0.0;
    [loop]
    for(;;)
    {
        float sum_1 = sum_0 + float(i_0);
        _S1 = sum_1;
        int i_1 = i_0 + int(1);
        if(i_1 < int(100))
        {
            i_0 = i_1;
            sum_0 = sum_1;
        }
        else
        {
            break;
        }
    }
    float _S2 = 0.0;
    int j_0 = int(0);
    sum_0 = _S1;
    [unroll]
    for(;;)
    {
        float sum_2 = sum_0 + float(j_0);
        _S2 = sum_2;
        int j_1 = j_0 + int(1);
        if(j_1 < int(100))
        {
            j_0 = j_1;
            sum_0 = sum_2;
        }
        else
        {
            break;
        }
    }
    return float4(_S2, 0.0, 0.0, 0.0);
}

