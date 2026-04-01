//TEST:SIMPLE(filecheck=CHECK): -target hlsl -line-directive-mode none

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

groupshared float s_shared;

[BackwardDifferentiable]
[PreferRecompute]
float get_thread_5_value(float v, uint group_thread_id)
{
    if(group_thread_id == 5)
    {
        s_shared = detach(v);
        // CHECK: tests/autodiff/warn-on-prefer-recompute-side-effects.slang(10): warning 42050: get_thread_5_value has [PreferRecompute] and may have side effects. side effects may execute multiple times. use [PreferRecompute(SideEffectBehavior.Allow)], or mark function with [__NoSideEffect]
        // CHECK: float get_thread_5_value(float v, uint group_thread_id)
        // CHECK:       ^~~~~~~~~~~~~~~~~~
    }
    GroupMemoryBarrierWithGroupSync();
    return s_shared;
}

[BackwardDifferentiable]
[PreferRecompute(SideEffectBehavior.Allow)] // Suppress warning here
float get_thread_6_value(float v, uint group_thread_id)
{
    if (group_thread_id == 6)
    {
        s_shared = detach(v);
        // CHECK-NOT: warning 42050

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
    bwd_diff(get_thread_6_value)(value, group_thread_id.x, 1.0f);

    outputBuffer[dispatch_thread_id.x] = value.d;
}