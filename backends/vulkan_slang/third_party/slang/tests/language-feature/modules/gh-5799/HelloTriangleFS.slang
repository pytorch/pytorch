//TEST:SIMPLE(filecheck=CHECK): -target spirv -I $dirname

//CHECK: OpEntryPoint

// shaders.slang

//
// This file provides a simple vertex and fragment shader that can be compiled
// using Slang. This code should also be valid as HLSL, and thus it does not
// use any of the new language features supported by Slang.
//

import Scene.Scene;

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 color;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};


// Fragment Shader

[shader("fragment")]
float4 main(
    CoarseVertex coarseVertex : CoarseVertex) : SV_Target
{
    float3 fragColor = coarseVertex.color;

    return float4(fragColor, 1.0);
}
