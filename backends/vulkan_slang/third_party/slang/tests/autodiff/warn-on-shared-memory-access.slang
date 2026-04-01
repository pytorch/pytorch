//TEST:SIMPLE(filecheck=CHECK): -target hlsl -line-directive-mode none

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

groupshared float s_shared;

[BackwardDifferentiable]
float get_thread_5_value(float v, uint group_thread_id)
{
    if(group_thread_id == 5)
    {
        // Using 'detach(v)' makes the error go away
        s_shared = v;
        // CHECK: tests/autodiff/warn-on-shared-memory-access.slang(14): error 41024: derivative is lost during assignment to non-differentiable location, use 'detach()' to clarify intention.
        // CHECK: s_shared = v;
        // CHECK:          ^
    }
    GroupMemoryBarrierWithGroupSync();
    return s_shared;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void computeMain(uint3 group_thread_id: SV_GroupThreadID, uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    DifferentialPair<float> value = diffPair(3.f, 0.f);

    bwd_diff(get_thread_5_value)(value, group_thread_id.x, 1.0f);

    outputBuffer[dispatch_thread_id.x] = value.d;
}