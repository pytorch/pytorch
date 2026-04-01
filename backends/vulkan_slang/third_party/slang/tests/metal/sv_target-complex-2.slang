//TEST:SIMPLE(filecheck=METAL): -target metal
//TEST:SIMPLE(filecheck=METAL_ASM): -target metallib
//TEST:SIMPLE(filecheck=WGSL): -target wgsl
//TEST:SIMPLE(filecheck=WGSLSPIRV): -target wgsl-spirv-asm

struct NestedReturn
{
    float4 debug1;
    float4 debug2;
};

struct Output
{
    float4 Diffuse : SV_Target0;
    NestedReturn val : SV_Target1;
    float4 Material : SV_Target2;
}

// METAL_ASM: define {{.*}} @fragmentMain
// METAL: color(0)
// METAL: color(1)
// METAL-DAG: color(3)
// METAL-DAG: color(2)

// WGSLSPIRV: %fragmentMain = OpFunction
// WGSL: @location(0) Diffuse
// WGSL: @location(1) debug1
// WGSL-DAG: @location(3) debug2
// WGSL-DAG: @location(2) Material

[shader("fragment")]
Output fragmentMain()
{
    return { float4(1), {float4(2), float4(2)}, float4(3) };
}
