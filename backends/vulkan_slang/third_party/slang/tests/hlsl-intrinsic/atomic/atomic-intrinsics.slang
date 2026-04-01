//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=DX12):-slang -compute -dx12 -profile cs_6_0 -use-dxil -shaderobj -output-using-type -xslang -DDX12
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=VK):-vk -emit-spirv-directly -compute -shaderobj -output-using-type -render-feature hardware-device -xslang -DVK
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=VK):-vk -emit-spirv-via-glsl -compute -shaderobj -output-using-type -render-feature hardware-device -xslang -DVK -xslang -minimum-slang-optimization
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=VK):-vk -emit-spirv-via-glsl -compute -shaderobj -output-using-type -render-feature hardware-device -xslang -DVK
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
void computeMain(uint groupIndex : SV_GroupIndex, int3 dispatchThreadID: SV_DispatchThreadID)
{
    shareMemUI = { 0, 0, 0, 0 };
    shareMemI = { 0, 0, 0, 0 };
    AllMemoryBarrierWithGroupSync();
    int idx = dispatchThreadID.x;
    float val = 0.0f;

    // InterlockedAdd
    InterlockedAdd(shareMemUI[idx], 1);
    val += shareMemUI[idx];

    InterlockedAdd(shareMemI[idx],  2);
    val += shareMemI[idx];

    InterlockedAdd(uintBuffer[idx], 1);
    val += uintBuffer[idx];

    InterlockedAdd(intBuffer[idx], 2);
    val += intBuffer[idx];

    InterlockedAdd(shareMemI[idx], -1);
    val += shareMemI[idx];

    InterlockedAdd(intBuffer[idx], -1);
    val += intBuffer[idx];

    // InterlockedAdd - original_value
    uint origui = 0;
    InterlockedAdd(shareMemUI[idx], 1, origui);
    val += shareMemUI[idx];
    val += origui;

    int origi = 0;
    InterlockedAdd(shareMemI[idx], 2, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedAdd(uintBuffer[idx], 1, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedAdd(intBuffer[idx], 2, origi);
    val += intBuffer[idx];
    val += origi;

    InterlockedAdd(shareMemI[idx], -1, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedAdd(intBuffer[idx], -1, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedAnd
    InterlockedAnd(shareMemUI[idx], 255);
    val += shareMemUI[idx];

    InterlockedAnd(shareMemI[idx], 255);
    val += shareMemI[idx];

    InterlockedAnd(uintBuffer[idx], 255);
    val += uintBuffer[idx];

    InterlockedAnd(intBuffer[idx], 255);
    val += intBuffer[idx];

    // InterlockedAnd - original_value
    InterlockedAnd(shareMemUI[idx], 255, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedAnd(shareMemI[idx], 255, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedAnd(uintBuffer[idx], 255, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedAnd(intBuffer[idx], 255, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedCompareExchange
    InterlockedCompareExchange(shareMemUI[idx], 1, 0, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedCompareExchange(shareMemI[idx], 1, 0, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedCompareExchange(uintBuffer[idx], 1, 0, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedCompareExchange(intBuffer[idx], 1, 0, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedCompareStore
    InterlockedCompareStore(shareMemUI[idx], 255, 0);
    val += shareMemUI[idx];

    InterlockedCompareStore(shareMemI[idx], 255, 0);
    val += shareMemI[idx];

    InterlockedCompareStore(uintBuffer[idx], 255, 0);
    val += uintBuffer[idx];

    InterlockedCompareStore(intBuffer[idx], 255, 0);
    val += intBuffer[idx];

    // InterlockedExchange
    InterlockedExchange(shareMemUI[idx], 1, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedExchange(shareMemI[idx], 1, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedExchange(uintBuffer[idx], 1, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedExchange(intBuffer[idx], 1, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedMax
    InterlockedMax(shareMemUI[idx], 0);
    val += shareMemUI[idx];

    InterlockedMax(shareMemI[idx], 0);
    val += shareMemI[idx];

    InterlockedMax(uintBuffer[idx], 0);
    val += uintBuffer[idx];

    InterlockedMax(intBuffer[idx], 0);
    val += intBuffer[idx];

    // InterlockedMax - original_value
    InterlockedMax(shareMemUI[idx], 0, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedMax(shareMemI[idx], 0, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedMax(uintBuffer[idx], 0, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedMax(intBuffer[idx], 0, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedMin
    InterlockedMin(shareMemUI[idx], 0);
    val += shareMemUI[idx];

    InterlockedMin(shareMemI[idx], 0);
    val += shareMemI[idx];

    InterlockedMin(uintBuffer[idx], 0);
    val += uintBuffer[idx];

    InterlockedMin(intBuffer[idx], 0);
    val += intBuffer[idx];

    // InterlockedMin - original_value
    InterlockedMin(shareMemUI[idx], 0, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedMin(shareMemI[idx], 0, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedMin(uintBuffer[idx], 0, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedMin(intBuffer[idx], 0, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedOr
    InterlockedOr(shareMemUI[idx], 2);
    val += shareMemUI[idx];

    InterlockedOr(shareMemI[idx], 4);
    val += shareMemI[idx];

    InterlockedOr(uintBuffer[idx], 6);
    val += uintBuffer[idx];

    InterlockedOr(intBuffer[idx], 8);
    val += intBuffer[idx];

    // InterlockedOr - original_value
    InterlockedOr(shareMemUI[idx], 2, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedOr(shareMemI[idx], 4, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedOr(uintBuffer[idx], 6, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedOr(intBuffer[idx], 8, origi);
    val += intBuffer[idx];
    val += origi;

    // InterlockedXor
    InterlockedXor(shareMemUI[idx], 2);
    val += shareMemUI[idx];

    InterlockedXor(shareMemI[idx], 4);
    val += shareMemI[idx];

    InterlockedXor(uintBuffer[idx], 6);
    val += uintBuffer[idx];

    InterlockedXor(intBuffer[idx], 8);
    val += intBuffer[idx];

    // InterlockedXor - original_value
    InterlockedXor(shareMemUI[idx], 2, origui);
    val += shareMemUI[idx];
    val += origui;

    InterlockedXor(shareMemI[idx], 4, origi);
    val += shareMemI[idx];
    val += origi;

    InterlockedXor(uintBuffer[idx], 6, origui);
    val += uintBuffer[idx];
    val += origui;

    InterlockedXor(intBuffer[idx], 8, origi);
    val += intBuffer[idx];
    val += origi;

    outputBuffer[idx] = val;
}

// DX12: 184
// DX12: 207
// DX12: 230
// DX12: 253
// VK: 184
// VK: 207
// VK: 230
// VK: 253