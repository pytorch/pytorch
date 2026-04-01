//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -allow-glsl -profile sm_5_0
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target hlsl -entry main -allow-glsl -profile sm_5_0 -ignore-capabilities

//CHECK: error 36107: entrypoint 'main' uses features that are not available in 'fragment' stage for 'hlsl' target.
//CHECK_IGNORE_CAPS-NOT: error 36100
[shader("fragment")]
float4 main()
{
    memoryBarrierAtomicCounter();
    return float4(1.0f);
}
