//TEST:SIMPLE(filecheck=HLSL): -target hlsl -stage compute -entry computeMain
//TEST:SIMPLE(filecheck=GLSL): -target glsl -stage compute -entry computeMain

RWStructuredBuffer<uint> outputBuffer;

// GLSL: error 30623
// HLSL: error 30623
groupshared uint globalMem = 1;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = globalMem ;
}
