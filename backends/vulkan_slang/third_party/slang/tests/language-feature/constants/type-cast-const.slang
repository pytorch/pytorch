//TEST(compute):COMPARE_COMPUTE: 
//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry computeMain -profile cs_5_0

int check<let b : bool>(int x)
{
    // CHECK: int v{{.*}}[int(2)]
    int v[((int)b) + 1];
    for (int i = 0; i < ((int)b) + 1; i++)
        v[i] = i;
 	return (int)v[x];
}

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal = tid;
    int outVal = check<true>(inVal + 1);
    outputBuffer[tid] = outVal;
}
