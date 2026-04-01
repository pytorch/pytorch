//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -compute -cpu -shaderobj
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHK):-slang -compute -dx11 -shaderobj -render-feature double
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHK):-slang -compute -dx12 -shaderobj
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -vk -compute -shaderobj -render-feature double
//TEST(compute):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-slang -cuda -compute -shaderobj
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHK):-slang -shaderobj -mtl

//CHK: 1

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // uint variant
    double d = double(dispatchThreadID.x);

    uint high;
    uint low;

    asuint(d, low, high);

    double r = asdouble(low, high);

    // uint2 variant
    double a = double(dispatchThreadID.x + 1);
    double b = double(dispatchThreadID.x + 2);

    uint2 high2;
    uint2 low2;

    asuint(a, low2.x, high2.x);
    asuint(b, low2.y, high2.y);

    double2 s = asdouble(low2, high2);

    // status
    outputBuffer[0] = (int(r) == 0)
        && (int(s.x) == 1) 
        && (int(s.y) == 2);
}
