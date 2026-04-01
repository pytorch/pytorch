//TEST:SIMPLE(filecheck=CHK):-target cpp -stage compute -entry computeMain

//CHK:struct Sampler2D
//CHK-NEXT:{
//CHK-NEXT:    Texture2D<{{.*}}> [[TEX:texture_[0-9]*]]
//CHK-NEXT:    SamplerState [[SMP:sampler_[0-9]*]]

Sampler2D<float4> s2D;

RWStructuredBuffer<float4> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    //CHK:[[VAR:[A-Za-z_][A-Za-z_0-9]*]].[[TEX]].Sample([[VAR]].[[SMP]]
    outputBuffer[0] = s2D.Sample(float2(0.5f, 0.5f));
}
