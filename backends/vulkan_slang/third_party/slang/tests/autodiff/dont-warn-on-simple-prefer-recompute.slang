//TEST:SIMPLE(filecheck=CHECK): -target hlsl -line-directive-mode none -stage compute -entry computeMain

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

[BackwardDifferentiable]
[PreferRecompute]
float comp(float a, float b)
{
    // CHECK: standard error = {
    // CHECK-NEXT: }
    return a * b;
}


[shader("compute")]
[numthreads(128, 1, 1)]
void computeMain(uint3 group_thread_id: SV_GroupThreadID, uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    DifferentialPair<float> value1 = diffPair(3.f, 0.f);
    DifferentialPair<float> value2 = diffPair(3.f, 0.f);

    bwd_diff(comp)(value1, value2, 1.f);

    outputBuffer[dispatch_thread_id.x] = value1.d;
}