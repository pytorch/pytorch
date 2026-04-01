// realtime-clock.slang

// TODO(JS):
// Disable FXC DX11/DX12 because currently FXC can't compile nvHLSLExtns.h 

//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-dx11 -slang -compute  -output-using-type -nvapi-slot u0 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -render-feature realtime-clock -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-d3d12 -compute -output-using-type -nvapi-slot u0 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-d3d12 -compute -use-dxil  -output-using-type -nvapi-slot u0 -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -output-using-type -shaderobj

// The test doesn't directly use this, but having this defined makes the 0 slot available if NVAPI is going to be used
// Only strictly necessary on the D3D11/D3D12 paths
//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):name=nvapiBuffer
RWStructuredBuffer<int> nvapiBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    
    uint ticksLow = getRealtimeClockLow();
    
    uint2 ticks = getRealtimeClock();
    
    outputBuffer[idx] = int(idx + ((ticksLow ^ ticks.x) ^ (ticks.x ^ ticksLow)));
}