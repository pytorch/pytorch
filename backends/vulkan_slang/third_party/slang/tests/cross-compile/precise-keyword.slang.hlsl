// precise-keyword.slang.hlsl
//TEST_IGNORE_FILE:

float4 main(float2 v_0 : V) : SV_TARGET
{
    float _S1 = v_0.x;
    precise float z_0;

    if (_S1 > 0.00000000000000000000)
    {
        z_0 = _S1 * v_0.y + _S1;
    }
    else
    {
        float _S2 = v_0.y;
        z_0 = _S2 * _S1 + _S2;
    }

    return (float4) z_0;
}