// loop-attribs.slang
// Test that loop attributes are correctly emitted to the resulting HLSL.

//TEST:CROSS_COMPILE:-target dxil-assembly  -entry main -stage fragment -profile sm_6_0 -loop-inversion

float4 main() : SV_Target
{
    float sum = 0.0f;
    
    [loop]
    for (int i = 0; i < 100; i++)
        sum += float(i);

    [unroll(10)]
    for (int j = 0; j < 100; j++)
        sum += float(j);

    return float4(sum, 0, 0, 0);
}
