//TEST:SIMPLE(filecheck=CHECK): -stage compute -entry computeMain -target hlsl
//CHECK: computeMain

//TEST_INPUT:ubuffer(data=[9 9 9 9], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    vector<float32_t, 4> vvv = vector<float32_t, 4>(0);
    float32_t ret = 0.0f;
    if (vvv.y < 1.0f)
    {
        ret = 1.0f;
    }
    else
    {
        if (vvv.y > 1.0f && outputBuffer[3] == 3)
        {
            ret = 0.0f;
        } else {
            if (true) {}
        }
    }
    outputBuffer[int(dispatchThreadID.x)] = int(ret);
}
