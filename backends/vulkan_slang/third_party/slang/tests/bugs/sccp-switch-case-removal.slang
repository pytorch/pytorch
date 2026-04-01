//TEST(compute,vulkan):COMPARE_COMPUTE_EX:-vk -slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

static const uint kConstant = 5;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float x = 1.0;
    switch(kConstant)
    {
    case 5:
        int tid = int(dispatchThreadID.x);
	    outputBuffer[tid] = 0;
        break;
    case 1:
        ddx(x); // this should trigger glslang error if it doesn't get eliminated by sccp.
        break;
    default:
        ddy(x); // this should trigger glslang error if it doesn't get eliminated by sccp.
        break;
    }
}