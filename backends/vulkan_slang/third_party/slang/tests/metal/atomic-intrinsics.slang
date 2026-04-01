//TEST:SIMPLE(filecheck=MTL):-target metal -entry computeMain -stage compute -DMETAL
//TEST:SIMPLE(filecheck=LIB):-target metallib -entry computeMain -stage compute -DMETAL
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -shaderobj -output-using-type

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -output-using-type
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -output-using-type


//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):name=uintBuffer
RWStructuredBuffer<uint> uintBuffer;
//TEST_INPUT:ubuffer(data=[0 1 2 3], stride=4):name=intBuffer
RWStructuredBuffer<int> intBuffer;

groupshared uint shareMemUI[4];
groupshared int shareMemI[4];

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint groupIndex : SV_GroupIndex)
{
    if (groupIndex == 0)
    {
        for (int i = 0; i < 4; ++i)
        {
            shareMemUI[i] = 0U;
            shareMemI[i] = 0;
        }
    }
    AllMemoryBarrierWithGroupSync();
    
    int idx = groupIndex;
    float val = 0.0f;

    // InterlockedAdd
    //MTL: atomic_uint threadgroup*{{.*}}shareMemUI
    //LIB: call {{.*}}.atomic.local.add.u.i32
    InterlockedAdd(shareMemUI[idx], uint(1));
    val += shareMemUI[idx];

    //MTL: atomic_int threadgroup*{{.*}}shareMemI
    //LIB: call {{.*}}.atomic.local.add.s.i32
    InterlockedAdd(shareMemI[idx],  2);
    val += shareMemI[idx];

    //MTL: atomic_uint device*{{.*}}uintBuffer
    //LIB: call {{.*}}.atomic.global.add.u.i32
    InterlockedAdd(uintBuffer[idx], 1);
    val += uintBuffer[idx];

    //MTL: atomic_int device*{{.*}}intBuffer
    //LIB: call {{.*}}.atomic.global.add.s.i32
    InterlockedAdd(intBuffer[idx], 2);
    val += intBuffer[idx];

    //LIB: call {{.*}}.atomic.local.add.s.i32
    InterlockedAdd(shareMemI[idx], -1);
    val += shareMemI[idx];

    //LIB: call {{.*}}.atomic.global.add.s.i32
    InterlockedAdd(intBuffer[idx], -1);
    val += intBuffer[idx];

    // InterlockedAdd - original_value
    uint origui = 0;
    //LIB: call {{.*}}.atomic.local.add.u.i32
    InterlockedAdd(shareMemUI[idx], 1, origui);
    val += shareMemUI[idx];
    val += origui;

    int origi = 0;
    //LIB: call {{.*}}.atomic.local.add.s.i32
    InterlockedAdd(shareMemI[idx], 2, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.add.u.i32
    InterlockedAdd(uintBuffer[idx], 1, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.add.s.i32
    InterlockedAdd(intBuffer[idx], 2, origi);
    val += intBuffer[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.local.add.s.i32
    InterlockedAdd(shareMemI[idx], -1, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.add.s.i32
    InterlockedAdd(intBuffer[idx], -1, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedAnd
    //LIB: call {{.*}}.atomic.local.and.u.i32
    InterlockedAnd(shareMemUI[idx], 255);
    val += shareMemUI[idx];

    //LIB: call {{.*}}.atomic.local.and.s.i32
    InterlockedAnd(shareMemI[idx], 255);
    val += shareMemI[idx];

    //LIB: call {{.*}}.atomic.global.and.u.i32
    InterlockedAnd(uintBuffer[idx], 255);
    val += uintBuffer[idx];

    //LIB: call {{.*}}.atomic.global.and.s.i32
    InterlockedAnd(intBuffer[idx], 255);
    val += intBuffer[idx];

    // InterlockedAnd - original_value
    //LIB: call {{.*}}.atomic.local.and.u.i32
    InterlockedAnd(shareMemUI[idx], 255, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.and.s.i32
    InterlockedAnd(shareMemI[idx], 255, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.and.u.i32
    InterlockedAnd(uintBuffer[idx], 255, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.and.s.i32
    InterlockedAnd(intBuffer[idx], 255, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedCompareExchange
    //LIB: call {{.*}}.atomic.local.cmpxchg.weak.i32
    InterlockedCompareExchange(shareMemUI[idx], 1, 0, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.cmpxchg.weak.i32
    InterlockedCompareExchange(shareMemI[idx], 1, 0, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.cmpxchg.weak.i32
    InterlockedCompareExchange(uintBuffer[idx], 1, 0, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.cmpxchg.weak.i32
    InterlockedCompareExchange(intBuffer[idx], 1, 0, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedCompareStore is not supported by Metal
#if !defined(METAL)
    InterlockedCompareStore(shareMemUI[idx], 255, 0);
    val += shareMemUI[idx];

    InterlockedCompareStore(shareMemI[idx], 255, 0);
    val += shareMemI[idx];

    InterlockedCompareStore(uintBuffer[idx], 255, 0);
    val += uintBuffer[idx];

    InterlockedCompareStore(intBuffer[idx], 255, 0);
    val += intBuffer[idx];
#endif

    // InterlockedExchange
    //LIB: call {{.*}}.atomic.local.xchg.i32
    InterlockedExchange(shareMemUI[idx], 1, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.xchg.i32
    InterlockedExchange(shareMemI[idx], 1, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.xchg.i32
    InterlockedExchange(uintBuffer[idx], 1, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.xchg.i32
    InterlockedExchange(intBuffer[idx], 1, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedMax
    //LIB: call {{.*}}.atomic.local.max.u.i32
    InterlockedMax(shareMemUI[idx], 0);
    val += shareMemUI[idx];

    //LIB: call {{.*}}.atomic.local.max.s.i32
    InterlockedMax(shareMemI[idx], 0);
    val += shareMemI[idx];

    //LIB: call {{.*}}.atomic.global.max.u.i32
    InterlockedMax(uintBuffer[idx], 0);
    val += uintBuffer[idx];

    //LIB: call {{.*}}.atomic.global.max.s.i32
    InterlockedMax(intBuffer[idx], 0);
    val += intBuffer[idx];

    // InterlockedMax - original_value
    //LIB: call {{.*}}.atomic.local.max.u.i32
    InterlockedMax(shareMemUI[idx], 0, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.max.s.i32
    InterlockedMax(shareMemI[idx], 0, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.max.u.i32
    InterlockedMax(uintBuffer[idx], 0, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.max.s.i32
    InterlockedMax(intBuffer[idx], 0, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedMin
    //LIB: call {{.*}}.atomic.local.min.u.i32
    InterlockedMin(shareMemUI[idx], 0);
    val += shareMemUI[idx];

    //LIB: call {{.*}}.atomic.local.min.s.i32
    InterlockedMin(shareMemI[idx], 0);
    val += shareMemI[idx];

    //LIB: call {{.*}}.atomic.global.min.u.i32
    InterlockedMin(uintBuffer[idx], 0);
    val += uintBuffer[idx];

    //LIB: call {{.*}}.atomic.global.min.s.i32
    InterlockedMin(intBuffer[idx], 0);
    val += intBuffer[idx];

    // InterlockedMin - original_value
    //LIB: call {{.*}}.atomic.local.min.u.i32
    InterlockedMin(shareMemUI[idx], 0, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.min.s.i32
    InterlockedMin(shareMemI[idx], 0, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.min.u.i32
    InterlockedMin(uintBuffer[idx], 0, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.min.s.i32
    InterlockedMin(intBuffer[idx], 0, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedOr
    //LIB: call {{.*}}.atomic.local.or.u.i32
    InterlockedOr(shareMemUI[idx], 2);
    val += shareMemUI[idx];

    //LIB: call {{.*}}.atomic.local.or.s.i32
    InterlockedOr(shareMemI[idx], 4);
    val += shareMemI[idx];

    //LIB: call {{.*}}.atomic.global.or.u.i32
    InterlockedOr(uintBuffer[idx], 6);
    val += uintBuffer[idx];

    //LIB: call {{.*}}.atomic.global.or.s.i32
    InterlockedOr(intBuffer[idx], 8);
    val += intBuffer[idx];

    // InterlockedOr - original_value
    //LIB: call {{.*}}.atomic.local.or.u.i32
    InterlockedOr(shareMemUI[idx], 2, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.or.s.i32
    InterlockedOr(shareMemI[idx], 4, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.or.u.i32
    InterlockedOr(uintBuffer[idx], 6, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.or.s.i32
    InterlockedOr(intBuffer[idx], 8, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedXor
    //LIB: call {{.*}}.atomic.local.xor.u.i32
    InterlockedXor(shareMemUI[idx], 2);
    val += shareMemUI[idx];

    //LIB: call {{.*}}.atomic.local.xor.s.i32
    InterlockedXor(shareMemI[idx], 4);
    val += shareMemI[idx];

    //LIB: call {{.*}}.atomic.global.xor.u.i32
    InterlockedXor(uintBuffer[idx], 6);
    val += uintBuffer[idx];

    //LIB: call {{.*}}.atomic.global.xor.s.i32
    InterlockedXor(intBuffer[idx], 8);
    val += intBuffer[idx];

    // InterlockedXor - original_value
    //LIB: call {{.*}}.atomic.local.xor.u.i32
    InterlockedXor(shareMemUI[idx], 2, origui);
    val += shareMemUI[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.local.xor.s.i32
    InterlockedXor(shareMemI[idx], 4, origi);
    val += shareMemI[idx];
    val += origi;

    //LIB: call {{.*}}.atomic.global.xor.u.i32
    InterlockedXor(uintBuffer[idx], 6, origui);
    val += uintBuffer[idx];
    val += origui;

    //LIB: call {{.*}}.atomic.global.xor.s.i32
    InterlockedXor(intBuffer[idx], 8, origi);
    val += intBuffer[idx];
    val += origi;

    outputBuffer[idx] = val;
}

// CHK: 184
// CHK: 207
// CHK: 230
// CHK: 253
