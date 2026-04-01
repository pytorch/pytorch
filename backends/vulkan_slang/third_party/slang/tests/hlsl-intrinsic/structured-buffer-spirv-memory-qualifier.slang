//TEST:SIMPLE(filecheck=CHECK_SPV):-stage fragment -entry main -target spirv -emit-spirv-directly

// CHECK_SPV: OpDecorate %x NonWritable
StructuredBuffer<uint> x;

// CHECK_SPV: OpDecorate %y NonWritable
ByteAddressBuffer y;

float4 main() : SV_Target
{
    return x[0] + y.Load<uint>(0);
}