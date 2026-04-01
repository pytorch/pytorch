//TEST:SIMPLE(filecheck=CHECK_SPIRV): -entry main -stage vertex -target spirv
//TEST:SIMPLE(filecheck=CHECK_GLSL): -entry main -stage vertex -target glsl
//TEST:SIMPLE(filecheck=CHECK_HLSL): -entry main -stage vertex -target hlsl
//TEST:SIMPLE(filecheck=CHECK_METAL): -entry main -stage vertex -target metal

#version 460

void main()
{
    float x = float(gl_VertexIndex + gl_BaseVertex) * 0.1f;
    float y = float(gl_InstanceIndex + gl_BaseInstance) * 0.2f;
    gl_Position = vec4(x, y, 0.0f, 1.0f); // Output 2D position with z=0 and w=1.

    // CHECK_SPIRV: BuiltIn BaseInstance
    // CHECK_GLSL: gl_BaseInstance
    // CHECK_HLSL: SV_StartInstanceLocation
    // CHECK_METAL: base_instance
}

