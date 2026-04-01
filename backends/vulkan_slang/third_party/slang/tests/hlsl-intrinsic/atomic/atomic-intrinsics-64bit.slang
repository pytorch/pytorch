//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=DX12):-slang -compute -dx12 -profile cs_6_6 -use-dxil -shaderobj -output-using-type

// This is to support 64-bit `Interlocked*` functions defined for HLSL SM6.6
// https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Int64_and_Float_Atomics.html

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):name=f32Buffer
RWStructuredBuffer<float> f32Buffer;
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=8):name=u64Buffer
RWStructuredBuffer<uint64_t> u64Buffer;
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=8):name=i64Buffer
RWStructuredBuffer<int64_t> i64Buffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]):name=fBuf
RWByteAddressBuffer fBuf;
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]):name=uBuf
RWByteAddressBuffer uBuf;
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]):name=iBuf
RWByteAddressBuffer iBuf;

groupshared float f32Shared[4];
groupshared uint64_t u64Shared[4];
groupshared int64_t i64Shared[4];
groupshared uint64_t indexAlloc;

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint groupIndex : SV_GroupIndex, int3 dispatchThreadID: SV_DispatchThreadID)
{
    f32Shared = { 0.f, 0.f, 0.f, 0.f };
    u64Shared = { 0, 0, 0, 0 };
    i64Shared = { 0, 0, 0, 0 };
    indexAlloc = 0;
    GroupMemoryBarrierWithGroupSync();

    int idx = dispatchThreadID.x;
    bool result = true;

    uint64_t u64Value[9];
    int64_t  i64Value[9];
    float    f32Value[9];

    // Add
    InterlockedAdd(u64Shared[idx], uint64_t(1));
    InterlockedAdd(i64Shared[idx],  int64_t(1));
    InterlockedAdd(u64Buffer[idx], uint64_t(1));
    InterlockedAdd(i64Buffer[idx],  int64_t(1));
    uBuf.InterlockedAdd64(idx * 8, uint64_t(1));
    iBuf.InterlockedAdd64(idx * 8,  int64_t(1));

    result = result
        && (u64Shared[idx] == 1)
        && (i64Shared[idx] == 1)
        && (u64Buffer[idx] == 1)
        && (i64Buffer[idx] == 1)
        && (uBuf.Load<uint64_t>(idx * 8) == 1)
        && (iBuf.Load< int64_t>(idx * 8) == 1)
        ;

    // Add - original_value
    InterlockedAdd(u64Shared[idx], uint64_t(1), u64Value[0]);
    InterlockedAdd(i64Shared[idx],  int64_t(1), i64Value[1]);
    InterlockedAdd(u64Buffer[idx], uint64_t(1), u64Value[2]);
    InterlockedAdd(i64Buffer[idx],  int64_t(1), i64Value[3]);
    uBuf.InterlockedAdd64(idx * 8, uint64_t(1), u64Value[4]);
    iBuf.InterlockedAdd64(idx * 8,  int64_t(1), i64Value[5]);

    result = result
        && (u64Value[0] == 1)
        && (i64Value[1] == 1)
        && (u64Value[2] == 1)
        && (i64Value[3] == 1)
        && (u64Value[4] == 1)
        && (i64Value[5] == 1)
        && (u64Shared[idx] == 2)
        && (i64Shared[idx] == 2)
        && (u64Buffer[idx] == 2)
        && (i64Buffer[idx] == 2)
        && (uBuf.Load<uint64_t>(idx * 8) == 2)
        && (iBuf.Load< int64_t>(idx * 8) == 2)
        ;

    // Bitwise-And
    InterlockedAnd(u64Shared[idx], uint64_t(3));
    InterlockedAnd(u64Buffer[idx], uint64_t(3));
    uBuf.InterlockedAnd64(idx * 8, uint64_t(3));

    result = result
        && (u64Shared[idx] == 2)
        && (u64Buffer[idx] == 2)
        && (uBuf.Load<uint64_t>(idx * 8) == 2)
        ;

    // And - original_value
    InterlockedAnd(u64Shared[idx], uint64_t(1), u64Value[0]);
    InterlockedAnd(u64Buffer[idx], uint64_t(1), u64Value[1]);
    uBuf.InterlockedAnd64(idx * 8, uint64_t(1), u64Value[2]);

    result = result
        && (u64Value[0] == 2)
        && (u64Value[1] == 2)
        && (u64Value[2] == 2)
        && (u64Shared[idx] == 0)
        && (u64Buffer[idx] == 0)
        && (uBuf.Load<uint64_t>(idx * 8) == 0)
        ;

    // Bitwise-Or
    InterlockedOr(u64Shared[idx], uint64_t(1));
    InterlockedOr(u64Buffer[idx], uint64_t(1));
    uBuf.InterlockedOr64(idx * 8, uint64_t(1));

    result = result
        && (u64Shared[idx] == 1)
        && (u64Buffer[idx] == 1)
        && (uBuf.Load<uint64_t>(idx * 8) == 1)
        ;

    // Or - original_value
    InterlockedOr(u64Shared[idx], uint64_t(2), u64Value[0]);
    InterlockedOr(u64Buffer[idx], uint64_t(2), u64Value[1]);
    uBuf.InterlockedOr64(idx * 8, uint64_t(2), u64Value[2]);

    result = result
        && (u64Value[0] == 1)
        && (u64Value[1] == 1)
        && (u64Value[2] == 1)
        && (u64Shared[idx] == 3)
        && (u64Buffer[idx] == 3)
        && (uBuf.Load<uint64_t>(idx * 8) == 3)
        ;

    // Bitwise-Xor
    InterlockedXor(u64Shared[idx], uint64_t(5));
    InterlockedXor(u64Buffer[idx], uint64_t(5));
    uBuf.InterlockedXor64(idx * 8, uint64_t(5));

    result = result
        && (u64Shared[idx] == 6)
        && (u64Buffer[idx] == 6)
        && (uBuf.Load<uint64_t>(idx * 8) == 6)
        ;

    // Xor - original_value
    InterlockedXor(u64Shared[idx], uint64_t(1), u64Value[0]);
    InterlockedXor(u64Buffer[idx], uint64_t(1), u64Value[1]);
    uBuf.InterlockedXor64(idx * 8, uint64_t(1), u64Value[2]);

    result = result
        && (u64Value[0] == 6)
        && (u64Value[1] == 6)
        && (u64Value[2] == 6)
        && (u64Shared[idx] == 7)
        && (u64Buffer[idx] == 7)
        && (uBuf.Load<uint64_t>(idx * 8) == 7)
        ;

    // Min
    InterlockedMin(u64Shared[idx], uint64_t(1));
    InterlockedMin(i64Shared[idx],  int64_t(1));
    InterlockedMin(u64Buffer[idx], uint64_t(1));
    InterlockedMin(i64Buffer[idx],  int64_t(1));
    uBuf.InterlockedMin64(idx * 8, uint64_t(1));
    iBuf.InterlockedMin64(idx * 8,  int64_t(1));

    result = result
        && (u64Shared[idx] == 1)
        && (i64Shared[idx] == 1)
        && (u64Buffer[idx] == 1)
        && (i64Buffer[idx] == 1)
        && (uBuf.Load<uint64_t>(idx * 8) == 1)
        && (iBuf.Load< int64_t>(idx * 8) == 1)
        ;

    // Min - original_value
    InterlockedMin(u64Shared[idx], uint64_t(2), u64Value[0]);
    InterlockedMin(i64Shared[idx],  int64_t(2), i64Value[1]);
    InterlockedMin(u64Buffer[idx], uint64_t(2), u64Value[2]);
    InterlockedMin(i64Buffer[idx],  int64_t(2), i64Value[3]);
    uBuf.InterlockedMin64(idx * 8, uint64_t(2), u64Value[4]);
    iBuf.InterlockedMin64(idx * 8,  int64_t(2), i64Value[5]);

    result = result
        && (u64Value[0] == 1)
        && (i64Value[1] == 1)
        && (u64Value[2] == 1)
        && (i64Value[3] == 1)
        && (u64Value[4] == 1)
        && (i64Value[5] == 1)
        && (u64Shared[idx] == 1)
        && (i64Shared[idx] == 1)
        && (u64Buffer[idx] == 1)
        && (i64Buffer[idx] == 1)
        && (uBuf.Load<uint64_t>(idx * 8) == 1)
        && (iBuf.Load< int64_t>(idx * 8) == 1)
        ;

    // Max
    InterlockedMax(u64Shared[idx], uint64_t(2));
    InterlockedMax(i64Shared[idx],  int64_t(2));
    InterlockedMax(u64Buffer[idx], uint64_t(2));
    InterlockedMax(i64Buffer[idx],  int64_t(2));
    uBuf.InterlockedMax64(idx * 8, uint64_t(2));
    iBuf.InterlockedMax64(idx * 8,  int64_t(2));

    result = result
        && (u64Shared[idx] == 2)
        && (i64Shared[idx] == 2)
        && (u64Buffer[idx] == 2)
        && (i64Buffer[idx] == 2)
        && (uBuf.Load<uint64_t>(idx * 8) == 2)
        && (iBuf.Load< int64_t>(idx * 8) == 2)
        ;

    // Max - original_value
    InterlockedMax(u64Shared[idx], uint64_t(0), u64Value[0]);
    InterlockedMax(i64Shared[idx],  int64_t(0), i64Value[1]);
    InterlockedMax(u64Buffer[idx], uint64_t(0), u64Value[2]);
    InterlockedMax(i64Buffer[idx],  int64_t(0), i64Value[3]);
    uBuf.InterlockedMax64(idx * 8, uint64_t(0), u64Value[4]);
    iBuf.InterlockedMax64(idx * 8,  int64_t(0), i64Value[5]);

    result = result
        && (u64Value[0] == 2)
        && (i64Value[1] == 2)
        && (u64Value[2] == 2)
        && (i64Value[3] == 2)
        && (u64Value[4] == 2)
        && (i64Value[5] == 2)
        && (u64Shared[idx] == 2)
        && (i64Shared[idx] == 2)
        && (u64Buffer[idx] == 2)
        && (i64Buffer[idx] == 2)
        && (uBuf.Load<uint64_t>(idx * 8) == 2)
        && (iBuf.Load< int64_t>(idx * 8) == 2)
        ;

    // Exchange
    InterlockedExchange(f32Shared[idx],    float(1), f32Value[0]);
    InterlockedExchange(u64Shared[idx], uint64_t(1), u64Value[1]);
    InterlockedExchange(i64Shared[idx],  int64_t(1), i64Value[2]);
    InterlockedExchange(f32Buffer[idx],    float(1), f32Value[3]);
    InterlockedExchange(u64Buffer[idx], uint64_t(1), u64Value[4]);
    InterlockedExchange(i64Buffer[idx],  int64_t(1), i64Value[5]);
    fBuf.InterlockedExchangeFloat(idx * 8, float(1), f32Value[6]);
    uBuf.InterlockedExchange64(idx * 8, uint64_t(1), u64Value[7]);
    iBuf.InterlockedExchange64(idx * 8,  int64_t(1), i64Value[8]);

    result = result
        && (f32Value[0] == 0)
        && (u64Value[1] == 2)
        && (i64Value[2] == 2)
        && (f32Value[3] == 0)
        && (u64Value[4] == 2)
        && (i64Value[5] == 2)
        && (f32Value[6] == 0)
        && (u64Value[7] == 2)
        && (i64Value[8] == 2)
        && (f32Buffer[idx] == 1.f)
        && (u64Shared[idx] == 1)
        && (i64Shared[idx] == 1)
        && (f32Buffer[idx] == 1.f)
        && (u64Buffer[idx] == 1)
        && (i64Buffer[idx] == 1)
        && (fBuf.Load<   float>(idx * 8) == 1.f)
        && (uBuf.Load<uint64_t>(idx * 8) == 1)
        && (iBuf.Load< int64_t>(idx * 8) == 1)
        ;

    // CompareStore
    InterlockedCompareStore(u64Shared[idx], uint64_t(1), uint64_t(0));
    InterlockedCompareStore(i64Shared[idx],  int64_t(1),  int64_t(0));
    InterlockedCompareStore(u64Buffer[idx], uint64_t(1), uint64_t(0));
    InterlockedCompareStore(i64Buffer[idx],  int64_t(1),  int64_t(0));
    uBuf.InterlockedCompareStore64(idx * 8, uint64_t(1), uint64_t(0));
    iBuf.InterlockedCompareStore64(idx * 8,  int64_t(1),  int64_t(0));

    result = result
        && (u64Shared[idx] == 0)
        && (i64Shared[idx] == 0)
        && (u64Buffer[idx] == 0)
        && (i64Buffer[idx] == 0)
        && (uBuf.Load<uint64_t>(idx * 8) == 0)
        && (iBuf.Load< int64_t>(idx * 8) == 0)
        ;

    // CompareStoreFloatBitwise
    InterlockedCompareStoreFloatBitwise(f32Shared[idx], float(1), float(0));
    InterlockedCompareStoreFloatBitwise(f32Buffer[idx], float(1), float(0));
    fBuf.InterlockedCompareStoreFloatBitwise(idx * 8,   float(1), float(0));

    result = result
        && (f32Shared[idx] == float(0))
        && (f32Buffer[idx] == float(0))
        && (fBuf.Load<float>(idx * 8) == float(0))
        ;

    // CompareExchange
    InterlockedCompareExchange(u64Shared[idx], uint64_t(0), uint64_t(1), u64Value[0]);
    InterlockedCompareExchange(i64Shared[idx],  int64_t(0),  int64_t(1), i64Value[1]);
    InterlockedCompareExchange(u64Buffer[idx], uint64_t(0), uint64_t(1), u64Value[2]);
    InterlockedCompareExchange(i64Buffer[idx],  int64_t(0),  int64_t(1), i64Value[3]);
    uBuf.InterlockedCompareExchange64(idx * 8, uint64_t(0), uint64_t(1), u64Value[4]);
    iBuf.InterlockedCompareExchange64(idx * 8,  int64_t(0),  int64_t(1), i64Value[5]);

    result = result
        && (u64Value[0] == 0)
        && (i64Value[1] == 0)
        && (u64Value[2] == 0)
        && (i64Value[3] == 0)
        && (u64Value[4] == 0)
        && (i64Value[5] == 0)
        && (u64Shared[idx] == 1)
        && (i64Shared[idx] == 1)
        && (u64Buffer[idx] == 1)
        && (i64Buffer[idx] == 1)
        && (uBuf.Load<uint64_t>(idx * 8) == 1)
        && (iBuf.Load< int64_t>(idx * 8) == 1)
        ;

    // CompareExchangeFloatBitwise
    InterlockedCompareExchangeFloatBitwise(f32Shared[idx], float(0), float(1), f32Value[0]);
    InterlockedCompareExchangeFloatBitwise(f32Buffer[idx], float(0), float(1), f32Value[1]);
    fBuf.InterlockedCompareExchangeFloatBitwise(idx * 8,   float(0), float(1), f32Value[2]);

    result = result
        && (f32Value[0] == float(0))
        && (f32Value[1] == float(0))
        && (f32Value[2] == float(0))
        && (f32Shared[idx] == float(1))
        && (f32Buffer[idx] == float(1))
        && (fBuf.Load<float>(idx * 8) == float(1))
        ;

    outputBuffer[idx] = int(result);
}

// DX12: 1
// DX12-NEXT: 1
// DX12-NEXT: 1
// DX12-NEXT: 1
