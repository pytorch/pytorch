//TEST:SIMPLE(filecheck=HLSL): -target hlsl -entry main
//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry main

// This test checks if `nointerpolation` is emitted more than once
//HLSL-NOT:nointerpolation nointerpolation 
//GLSL-NOT:flat flat

[shader("pixel")]
void main(out float4 SV_Target : SV_Target, const in nointerpolation float4 color : color)
{
    for (int i = 0; i < 10; ++i)
    {}

    SV_Target = color;
}
