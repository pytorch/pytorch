//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile glsl_450 -entry MainCs -stage compute  -fvk-b-shift 0 0 -fvk-s-shift 14 0 -fvk-t-shift 30 0 -fvk-u-shift 158 0


// CHECK-DAG: layout(std430, binding = 159) buffer  
// CHECK-DAG: } g_ByteBuffer

// CHECK-DAG: layout(std430, binding = 158) buffer  

RWStructuredBuffer<uint> g_OutputCullBits;
RWByteAddressBuffer g_ByteBuffer;

[numthreads(32, 1, 1)]
void MainCs(uint3 vThreadId : SV_DispatchThreadID, uint3 vGroupThreadId : SV_GroupThreadID, uint3 vGroupId : SV_GroupID)
{
    g_OutputCullBits[vThreadId.x] = g_ByteBuffer.Load(0);
}