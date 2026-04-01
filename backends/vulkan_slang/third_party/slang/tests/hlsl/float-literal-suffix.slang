//TEST:SIMPLE(filecheck=HLSL):-target hlsl -profile ps_6_6 -entry fragmentMain
//TEST:SIMPLE(filecheck=DXIL):-target dxil -profile ps_6_6 -entry fragmentMain

float4 fragmentMain(float2 uv : TEXCOORD) : SV_Target
{
    //HLSL:, ddx({{.*}}1.5f)
    //DXIL: = call float @dx.op.unary.f32(i32 {{.*}} ; DerivCoarseX(value)
    float val = 1.5;
    return float4(1.0, ddx(val), 0.0, 1.0);
}