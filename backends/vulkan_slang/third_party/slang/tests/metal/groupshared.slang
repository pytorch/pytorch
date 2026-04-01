//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib

uniform RWStructuredBuffer<float> outputBuffer;

struct MyBlock
{
    StructuredBuffer<float> b1;
    StructuredBuffer<float> b2;
}
ParameterBlock<MyBlock> block;

groupshared int myArr[16];

void func(float v)
{
    outputBuffer[0] = myArr[0];
}

// CHECK: array<int, int(16)> threadgroup* myArr{{.*}};
// CHECK: {{\[\[}}kernel{{\]\]}} void main_kernel
// CHECK: threadgroup array<int, int(16)> myArr{{.*}};
// CHECK: (&kernelContext{{.*}})->myArr{{.*}} = &myArr{{.*}};
// CHECK-ASM: define void @main_kernel

[numthreads(1,1,1)]
void main_kernel(uint3 tid: SV_DispatchThreadID)
{
    myArr[tid.x] = tid.x;
    func(3.0f);
}
