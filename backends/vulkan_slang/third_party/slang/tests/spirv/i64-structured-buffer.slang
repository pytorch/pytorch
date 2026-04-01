//TEST:SIMPLE(filecheck=CHECK): -stage compute -target spirv -emit-spirv-directly -entry main

RWStructuredBuffer<float> output;

// Check that 64bit integer index can be used in structured buffers without conversion to int.

// CHECK: %[[INDEX:[A-Za-z0-9_]+]] = OpLoad %long %{{.*}}
// CHECK: OpAccessChain %_ptr_StorageBuffer_float %output %int_0 %[[INDEX]]

uniform int64_t index;

[numthreads(1,1,1)]
void main()
{
    output[index] = 1;
}
