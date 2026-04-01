// gh-775.slang
//TEST(compute):COMPARE_COMPUTE: -shaderobj

int test(int inVal)
{
    float4x4 identity = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    float4 v = float4(inVal, inVal+1, inVal+2, inVal+3);

    v = mul(identity, v);

    return int(dot(v, float4(1, 16, 256, 4096)));
}

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
	uint tid = dispatchThreadID.x;

	int inVal = int(tid);
	int outVal = test(inVal);

	outputBuffer[tid] = outVal;
}