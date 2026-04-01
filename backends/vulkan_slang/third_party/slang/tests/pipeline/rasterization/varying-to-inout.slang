// Test passing a varying parameter direclty to an inout parameter.

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment

// CHECK: OpEntryPoint Fragment %main "main"
struct PS_IN
{
    float3 pos : SV_Position;
    float4 color : COLOR;
}

void test(inout PS_IN v)
{
    v.color = v.color + v.pos.x;
}

[shader("fragment")]
float4 main(PS_IN psIn):SV_Target
{
    test(psIn);
    return psIn.color;
}
