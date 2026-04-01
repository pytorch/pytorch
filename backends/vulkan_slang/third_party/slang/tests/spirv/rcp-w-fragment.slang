//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage fragment -fvk-use-dx-position-w -emit-spirv-directly

// CHECK: OpFDiv
float4 main(float4 pos : SV_Position) : SV_Target
{
    return pos;
}