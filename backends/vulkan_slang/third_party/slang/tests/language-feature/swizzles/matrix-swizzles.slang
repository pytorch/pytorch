// matrix-swizzle.slang

//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that matrix swizzle works correctly
// Matrix swizzles can either be one or zero indexed
// Reference: https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math

int test(int val)
{
    float2x2 worldMatrix = float2x2(val + 0, val + 1, val + 2, val + 3);
    float2 tempVector1;
    float2 tempVector2;

    // TODO: make left-hand side matrix swizzles work
    tempVector1 = worldMatrix._m00_m11;
    tempVector2 = worldMatrix._12_21;

    // return tempMatrix[0][0] + tempMatrix[0][1] = val + 0 + val + 1
    return int(tempVector1.x + tempVector2.x);
}


//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = test(inVal);
    outputBuffer[tid] = outVal;
}
