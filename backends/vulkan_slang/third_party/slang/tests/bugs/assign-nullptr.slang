//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry computeMain -O0

//CHECK: %[[NULLPTR_VAL:[a-zA-Z0-9_]+]] = OpConvertUToPtr %_ptr_PhysicalStorageBuffer_int %{{.*}}
//CHECK: OpStore %ptr %[[NULLPTR_VAL]]

[vk::push_constant] int* dest;
[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int* ptr = nullptr;
    if (dispatchThreadID.x % 2 == 0) ptr = dest;
    if (ptr) *ptr = 123;

}
