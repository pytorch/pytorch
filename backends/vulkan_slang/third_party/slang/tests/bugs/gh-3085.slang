//TEST:SIMPLE(filecheck=CHECK): -entry MainCs -stage compute -profile glsl_450 -target spirv
//CHECK: OpEntryPoint

RWTexture2D<float4> g_Test;

Texture2D<float4> g_inoutColorReadonly;

[ForceInline] float3 LoadSourceColor(uint2 pixelPos, constexpr int2 offset, int sampleIndex)
{
    float3 color = g_inoutColorReadonly.Load(int3(pixelPos, 0), offset).rgb;

    return color;
}

[numthreads(16, 16, 1)]
void MainCs(uint3 groupID: SV_GroupID, uint3 groupThreadID: SV_GroupThreadID)
{

    uint2 pixelPos = groupID.xy * int2((16 - 2), (16 - 2)) + groupThreadID.xy - int2(1, 1);
    float3 color = LoadSourceColor(pixelPos, int2(0, 0), 0).rgb;

    // Note this also doesn't work
    //[unroll]
    // for ( int i = 0; i < 3; i++ )
    //{
    // color+= LoadSourceColor ( pixelPos , int2 ( i%3 , i/3 ) , msaaSampleIndex ) . rgb ;
    //}
    g_Test[int2(pixelPos.x / 2, pixelPos.y + 0)] = float4(color, 1.0);
}
