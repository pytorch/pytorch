// gh-569.slang
//TEST(compute):COMPARE_COMPUTE: -shaderobj

// Test that correct scoping is used in generated HLSL/GLSL,
// even when dominator tree and structured control flow disagree.

uint test(uint inVal)
{
	uint tmp = inVal;
    for(;;)
    {
    	if(tmp < 4)
    	{
    		tmp++;
    	}
    	else
    	{
    		break;
    	}
    }
    return tmp + inVal;
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name gBuffer
RWStructuredBuffer<uint> gBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    uint val = tid;
    val = test(val);

    gBuffer[tid] = val;
}
