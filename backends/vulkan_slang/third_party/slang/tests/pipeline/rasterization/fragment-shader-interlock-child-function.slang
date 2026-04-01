//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment

RasterizerOrderedTexture2D<float4> texture;

//
// Tests that an interlock region place at a non-entry point function will properly apply the SPIRV execution mode
// to the entry point function and not the child function.
//

// CHECK: OpEntryPoint
// CHECK: OpExecutionMode %main PixelInterlockOrderedEXT

float4 foo(float4 coords)
{
    float4 result;

    beginInvocationInterlock();
    {
        result = texture[uint2(coords.xy)];
        texture[uint2(coords.xy)] = result + coords;
    }
    endInvocationInterlock();

    return result;
}

[shader("fragment")]
float4 main(float4 coords : COORDS) : SV_Target
{
    return foo(coords);
}
