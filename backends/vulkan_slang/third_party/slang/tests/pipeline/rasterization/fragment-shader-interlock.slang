// fragment-shader-interlock.slang

// Test that explicit use of fragment-shader interlock (FSI)
// operations is supported by Slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment
//TEST:SIMPLE(filecheck=CHECK):-emit-spirv-directly -target spirv -entry main -stage fragment


// CHECK: OpCapability FragmentShaderPixelInterlockEXT
// CHECK: OpExtension "SPV_EXT_fragment_shader_interlock"
// CHECK: OpEntryPoint Fragment %main "main"

// CHECK-DAG: OpExecutionMode %main OriginUpperLeft

// CHECK-DAG: OpExecutionMode %main PixelInterlockOrderedEXT

[shader("fragment")]
void main(
    float4 coords : COORDS,
    uniform RasterizerOrderedTexture2D<float4> texture,
    out float4 result : SV_Target)
{
    beginInvocationInterlock();

    result = texture[uint2(coords.xy)];
    texture[uint2(coords.xy)] = result + coords;

    endInvocationInterlock();
}
