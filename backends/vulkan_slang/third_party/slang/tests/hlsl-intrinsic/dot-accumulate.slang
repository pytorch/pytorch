//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
// Does not run on DX11 as SM 6.4 is required.
//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx11
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -dx12 -profile cs_6_4 -use-dxil -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-metal -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-wgsl -compute -shaderobj -render-feature half -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute -shaderobj -g0 -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint outputIndex = 0;

    //
    // dot4add_u8packed()
    // [4 3 2 1]  dot [1 2 4 2] + 5
    // (4 * 1) + (3 * 2) + (2 * 4) + (1 * 2) + 5 = 25
    //
    uint unsignedX = 0x01020304U;
    uint unsignedY = 0x02040201U;
    uint unsignedAcc = 5U; 
    uint unsignedResult = dot4add_u8packed(unsignedX, unsignedY, unsignedAcc);
    outputBuffer[outputIndex++] = unsignedResult;

    //
    // dot4add_i8packed()
    // [6 2 3 -1] dot [-2 -6 2 6] - 100
    // (6 * -2) + (2 * -6) + (3 * 2) + (-1 * 6) - 100 = -124
    //
    int signedX = 0xFF030206;
    int signedY = 0x0602FAFE;
    int signedAcc = -100;
    int signedResult = dot4add_i8packed(signedX, signedY, signedAcc);
    outputBuffer[outputIndex++] = signedResult;

    //
    // dot2add()
    // [10.8 -3.3] dot [1.4 -20.3] - 2.11 
    // (10.8 * 1.4) + (-3.3 * -20.3) - 2.0 = 80.11
    //
    half2 half2X = half2(half(10.8), half(-3.3));
    half2 half2Y = half2(half(1.4), half(-20.3));

    // `half2Acc` is assigned -2.0 here.
    // Thread index is used so that `half2Acc` will not be implicitly emitted as literal `-2.0` which
    // may be treated as a double by DXC and cause it to fail to compile because no overload exists for `dot2add` that
    // accepts double.
    float half2Acc = float(dispatchThreadID.x + 1) * -2.0f;
    float half2Result = dot2add(half2X, half2Y, half2Acc);
    outputBuffer[outputIndex++] = int(half2Result);
}
