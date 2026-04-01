//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

// Print an error when the numLanes is an invalid value for WaveSize.
// The value has to be a power of 2 between 4 and 128, inclusive.
// In other words, the set: [4, 8, 16, 32, 64, 128].

// "5" is an invalid value for WaveSize
// CHECK: error 31103:
[WaveSize(5)]
[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
	int tid = dispatchThreadID.x;
	outputBuffer[tid] = tid;
}
