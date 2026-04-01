//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpPtrAccessChain

struct EmptyStruct {
};

[vk::push_constant] EmptyStruct* pc;

RWStructuredBuffer<int> outputBuffer;

[numthreads(64)]
void ComputeMain(uint tid: SV_DispatchThreadID) {
    outputBuffer[tid] = ((int*)(pc))[0];
}