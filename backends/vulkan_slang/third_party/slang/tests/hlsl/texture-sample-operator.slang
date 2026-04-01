//TEST:SIMPLE(filecheck=HLSL): -target hlsl -stage vertex
//TEST:SIMPLE(filecheck=GLSL): -target glsl -stage vertex
//TEST:SIMPLE(filecheck=METAL): -target metal -stage vertex
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -stage vertex
//TEST:SIMPLE(filecheck=SPIRV): -target spirv-asm -stage vertex

Texture2DMS <int3> t1;
Texture2DMSArray <float4> t2;

float4 main()
{
    // HLSL: .sample
    // GLSL: texelFetch
    // SPIRV: OpImageFetch %v4int {{.*}} Sample {{.*}}
    // METAL: .read
    uint2 p1 = uint2(1, 2);
    int3 a1 = t1.sample[7][p1];

    // HLSL: .sample
    // GLSL: texelFetch
    // SPIRV: OpImageFetch %v4float {{.*}} Sample {{.*}}
    // METAL: .read
    uint p2 = uint(1);
    float4 a2 = t2.sample[p2][uint3(1, 2, 3)];

    return float4(float3(a1), 0) + a2;
}
