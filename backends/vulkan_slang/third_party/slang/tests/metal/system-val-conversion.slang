//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib
//TEST:SIMPLE(filecheck=CHECK): -target metal

// Test that we always emit correct type for system value and insert conversion logic
// if the declared type of the SV is different from the spec-defined type.

uniform RWStructuredBuffer<float> outputBuffer;

RWByteAddressBuffer buffer;

// CHECK-ASM: define void @main_kernel

struct TestStruct
{
    uint8_t a;
    float16_t h;
    float b;
    float4 c;
    float4x3 d;
}

// CHECK: void main_kernel(uint3 tid{{.*}}
// CHECK: int tid{{.*}} = int(tid{{.*}}.x);

[numthreads(1,1,1)]
void main_kernel(int tid: SV_DispatchThreadID)
{
    buffer.Store(128, buffer.Load<TestStruct>(tid));
}
