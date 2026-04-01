//TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -allow-glsl -profile sm_5_0

//CHECK: main
[shader("fragment")]
float4 main()
{
    return float4(1.0f);
}
