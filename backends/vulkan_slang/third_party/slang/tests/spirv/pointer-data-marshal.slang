//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

struct Foo {
  column_major float3x2 m;
};

struct Params {
  Foo *foo;
};

// CHECK: %[[PTR0:[A-Za-z0-9_]+]] = OpAccessChain %_ptr_PhysicalStorageBuffer__arr_v3float_int_2 %{{.*}} %int_0
// CHECK: %[[PTR1:[A-Za-z0-9_]+]]  = OpAccessChain %_ptr_PhysicalStorageBuffer_v3float %[[PTR0]] %int_1
// CHECK: %[[PTR2:[A-Za-z0-9_]+]]  = OpAccessChain %_ptr_PhysicalStorageBuffer_float %[[PTR1]] %int_2

ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(1,1,1)]
void main() {
  params.foo.m[2][1] += 1.0;
}