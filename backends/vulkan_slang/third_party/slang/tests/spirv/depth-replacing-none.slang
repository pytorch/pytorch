//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage fragment -entry main -emit-spirv-directly

// Test that a fragment shader that does not modify depth does not
// have a DepthReplacing decoration.

struct PSOut
{
    float4 color : SV_Target;
}
// CHECK-NOT: OpExecutionMode {{.*}} DepthReplacing

// Using SV_Position does not cause us generate DepthReplacing.
PSOut main(float4 pos : SV_Position)
{
    PSOut psout;
    psout.color = pos;
    return psout;
}
