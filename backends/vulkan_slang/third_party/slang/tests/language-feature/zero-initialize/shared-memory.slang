//DISABLE_TEST:SIMPLE(filecheck=HLSL): -target hlsl -stage compute -entry computeMain -zero-initialize
//DISABLE_TEST:SIMPLE(filecheck=GLSL): -target glsl -stage compute -entry computeMain -zero-initialize

RWStructuredBuffer<uint> outputBuffer;

// GLSL-NOT: error 30623
// HLSL-NOT: error 30623

// GLSL-NOT: globalMem{{.*}} =
// HLSL-NOT: globalMem{{.*}} =

groupshared uint globalMem;

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    outputBuffer[0] = globalMem;
}
