//TEST:SIMPLE(filecheck=SPIRV): -target spirv -fvk-use-entrypoint-name
//TEST:SIMPLE(filecheck=METAL): -target metal
//TEST:SIMPLE(filecheck=METALASM): -target metallib


// METAL-DAG: {{.*}}{{\[\[}}raster_order_group(0){{\]\]}} {{\[\[}}buffer(0)
// METAL-DAG: {{.*}}{{\[\[}}raster_order_group(0){{\]\]}} {{\[\[}}texture(0)
// METAL-DAG: {{.*}}{{\[\[}}raster_order_group(0){{\]\]}} {{\[\[}}buffer(1)

// METALASM: @fragMain

RasterizerOrderedByteAddressBuffer buffer;

[shader("fragment")]
float4 fragMain() : SV_Target
{
    // SPIRV: %fragMain{{.*}} = OpFunction
    // SPIRV: OpBeginInvocationInterlockEXT
    // SPIRV: OpEndInvocationInterlockEXT

    buffer.Store(0, 0x12345678);
    return float4(1, 0, 0, 1);
}

RasterizerOrderedTexture2D tex;

[shader("fragment")]
float4 fragMain2() : SV_Target
{
    // SPIRV: %fragMain2{{.*}} = OpFunction
    // SPIRV: OpBeginInvocationInterlockEXT
    // SPIRV: OpEndInvocationInterlockEXT

    tex[uint2(0, 0)] = float4(1, 0, 0, 1);
    return float4(1, 0, 0, 1);
}

RasterizerOrderedStructuredBuffer<float> buffer2;

[shader("fragment")]
float4 fragMain3() : SV_Target
{
    // SPIRV: %fragMain3{{.*}} = OpFunction
    // SPIRV: OpBeginInvocationInterlockEXT
    // SPIRV: OpEndInvocationInterlockEXT

    buffer2[0] = 1;
    return float4(1, 0, 0, 1);
}

