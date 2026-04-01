
// Emit 0xFFFF as the stride value for the unsized array

//TEST:SIMPLE(filecheck=CHECK): -entry fragment -stage fragment -emit-spirv-directly -target spirv-assembly -emit-spirv-directly
struct Descriptors {
  uint count;
  uint4 array[];
}

struct Context {
  Descriptors *descriptors;
}

[[vk::binding(0)]] ConstantBuffer<Context> context;

// Dummy entry point.
[shader("fragment")]
float4 fragment(): SV_Target
{
  return float4(float(context.descriptors->array[0].x), 1., 1., 1.);
}

// CHECK: OpDecorate %_ptr_PhysicalStorageBuffer_Descriptors_natural ArrayStride 4
// CHECK: OpBitcast %ulong
// CHECK: OpIAdd %ulong %{{.*}} %ulong_4
// CHECK: OpBitcast %_ptr_PhysicalStorageBuffer