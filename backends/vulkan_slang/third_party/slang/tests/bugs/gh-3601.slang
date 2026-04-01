//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -O0
struct TestStruct
{
    uint index;
};

[[vk::binding(2, 0)]] StructuredBuffer<uint64_t> test;

struct PP
{
    int data;
    int data2;
}
struct Data
{
    int data;
    PP* pNext;
};

void funcThatTakesPointer(PP* p)
{
    p.data = 2;
}
int* funcThatReturnsPointer(PP* p)
{
    return &p.data;
}

// CHECK: OpEntryPoint

[[vk::binding(0, 0)]] StructuredBuffer<Data> buffer;
[[vk::binding(1, 0)]] RWStructuredBuffer<int> output;
[shader("compute")]
[numthreads(8, 8, 1)]
void main(int id : SV_DispatchThreadID)
{
    TestStruct * ptr = (TestStruct *)(test[0]);
    output[0] = buffer[ptr.index].pNext.data;
    let pData = &(buffer[0].pNext.data);
    // CHECK: OpPtrAccessChain
    int* pData1 = pData + 1;
    *pData1 = 3;
    *(int2*)pData = int2(1, 2);
    pData1[-1] = 2;
    buffer[0].pNext[1] = {5};
    // CHECK: OpConvertPtrToU
    // CHECK: OpINotEqual
    if (pData1)
    {
        *(funcThatReturnsPointer(buffer[0].pNext)) = 4;
    }
    if (pData1 > pData)
    {
        funcThatTakesPointer(buffer[0].pNext);
    }
}