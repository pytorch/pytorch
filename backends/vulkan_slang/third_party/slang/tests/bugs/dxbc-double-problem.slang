//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -output-using-type -shaderobj -render-feature double
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -output-using-type -shaderobj -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -output-using-type -shaderobj
//TEST(compute):COMPARE_COMPUTE_EX:-dx12 -compute -use-dxil -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-dx12 -compute -output-using-type -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE:-slang -shaderobj -mtl
// Not supported in WGSL: Double and other unsupported scalar types
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-wgpu

// The problem this test shows is around handling of double with dxbc on D3D12. In that combination
// this code does not write the correct value into the first element - it appears as 0, where 
// clearly w * pi_180 where w = 1 means the answer is not zero.
//
// To demonstrate the problem, renable the -dx12 -compute -output-using-type test. It will output 0 for the first item.

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out, name outputBuffer
RWStructuredBuffer<double> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    double w = double(dispatchThreadID.x + 1) * 38.63323592724938038L;

    double pi = 3.14159274101257324L;

    double pi_180 = pi/180;
    double rad = w * pi_180;            
            
    outputBuffer[0] = rad;
    outputBuffer[1] = pi;
    outputBuffer[2] = pi_180;
    outputBuffer[3] = w;
}
