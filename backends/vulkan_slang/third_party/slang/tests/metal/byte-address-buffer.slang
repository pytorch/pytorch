//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib

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

[numthreads(1,1,1)]
void main_kernel(uint3 tid: SV_DispatchThreadID)
{
    // CHECK: uint [[WORD0:[a-zA-Z0-9_]+]] = as_type<uint>({{.*}}[(int(0))>>2]);
    // CHECK: uint8_t [[A:[a-zA-Z0-9_]+]] = uint8_t(([[WORD0]] >> 0U) & 255U);
    // CHECK: uint [[WORD1:[a-zA-Z0-9_]+]] = as_type<uint>({{.*}}[(int(0))>>2]);
    // CHECK: half [[H:[a-zA-Z0-9_]+]] = as_type<half>(ushort(([[WORD1]] >> 16U) & 65535U));

    // CHECK: {{.*}}[(int(128))>>2] = as_type<uint32_t>(({{.*}} & 4294967040U) | (uint([[A]]) << 0U));
    // CHECK: {{.*}}[(int(128))>>2] = as_type<uint32_t>(({{.*}} & 65535U) | (uint(as_type<ushort>([[H]])) << 16U));
    buffer.Store(128, buffer.Load<TestStruct>(0));
}
