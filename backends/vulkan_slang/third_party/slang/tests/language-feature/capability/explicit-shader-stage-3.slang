//TEST:SIMPLE(filecheck=CHECK): -target hlsl -stage vertex -entry main -allow-glsl -profile sm_5_0

//CHECK: warning 38006
[shader("fragment")]
float4 main()
{
    return float4(1.0f);
}
