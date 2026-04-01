//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage fragment -entry main -emit-spirv-directly

struct PSOut
{
    float4 color : SV_Target;
    float depth : SV_Depth;
}
// CHECK: OpExecutionMode {{.*}} DepthReplacing

PSOut main()
{
    PSOut psout;
    psout.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    psout.depth = 0.5f;
    return psout;
}
