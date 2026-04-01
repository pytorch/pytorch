//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile glsl_450 -entry main -stage compute
//TEST:SIMPLE(filecheck=SPV):-target spirv -profile glsl_450 -entry main -stage compute -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=SPV):-target spirv -profile glsl_450 -entry main -stage compute -emit-spirv-directly
// SPV: OpMemoryModel PhysicalStorageBuffer64
// SPV: OpEntryPoint GLCompute {{.*}} "main" {{.*}}
// SPV: OpTypePointer PhysicalStorageBuffer
// SPV: OpConvertPtrToU
// SPV: OpINotEqual
struct MyStruct
{
    float4 position;
    float4x4 transform;
}

// CHECK: layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer BufferPointer_MyStruct
// CHECK-NEXT: {
// CHECK-NEXT: MyStruct{{.*}} _data;
// CHECK-NEXT: }

// CHECK: struct ConstBufferPointer
// CHECK-NEXT: {
// CHECK-NEXT: BufferPointer_MyStruct{{.*}} _ptr
// CHECK-NEXT: }

struct Globals
{
    ConstBufferPointer<MyStruct> pStruct;
}

ConstantBuffer<Globals> gGlobals;

RWStructuredBuffer<uint> outputBuffer;

[numthreads(1,1,1)]
void main(int3 tid: SV_DispatchThreadID)
{
    if (gGlobals.pStruct.isValid())
    {
        MyStruct s = gGlobals.pStruct.get();
        outputBuffer[tid.x] = uint(s.position.x);
        MyStruct s1 = gGlobals.pStruct[3];
        outputBuffer[tid.x + 1] = uint(s1.position.x);
    }
}
