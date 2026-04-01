//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage compute -emit-spirv-directly

// CHECK: OpEntryPoint

ConstantBuffer<Ptr<int>> cbPtr;
void main(int id : SV_DispatchThreadID)
{
    // Check that the index operand is translated directly into a 64bit integer
    // in th resulting SPIR-V without any truncations.
    // CHECK: OpPtrAccessChain %_ptr_PhysicalStorageBuffer_int %{{.*}} %long_123
    cbPtr[123ll] = 4;
}
