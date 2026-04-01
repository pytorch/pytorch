//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry main -stage fragment -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target glsl -entry main -stage fragment -allow-glsl -ignore-capabilities
// CHECK_IGNORE_CAPS-NOT: error 36107
// CHECK: error 36107

float4 main()
{
    memoryBarrierAtomicCounter();
    return float4(1.0f);
}
