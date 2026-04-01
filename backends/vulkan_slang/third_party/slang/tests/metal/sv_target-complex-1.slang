//TEST:SIMPLE(filecheck=METAL): -target metal
//TEST:SIMPLE(filecheck=METAL_ASM): -target metallib
//TEST:SIMPLE(filecheck=WGSL): -target wgsl
//TEST:SIMPLE(filecheck=WGSLSPIRV): -target wgsl-spirv-asm

struct NestedReturn
{
    float4 debug;
};
struct NestedReturn2
{
    float4 debugAux1;
    float4 debugAux2;
};

// Semantics are supposed to ignore uppercase/lowercase differences
struct Output
{
    float4 Diffuse : SV_TarGet0;
    NestedReturn debug1 : SV_Target1;
    float4 Material : SV_TArgeT2;
    NestedReturn2 debug2 : SV_TaRget3;
}

// METAL_ASM: define {{.*}} @fragmentMain
// METAL: color(0)
// METAL-DAG: color(1)
// METAL-DAG: color(2)
// METAL-DAG: color(3)
// METAL-DAG: color(4)

// WGSLSPIRV: %fragmentMain = OpFunction
// WGSL: @location(0) Diffuse
// WGSL: @location(1) debug
// WGSL: @location(2) Material
// WGSL: @location(3) debugAux1
// WGSL: @location(4) debugAux2

[shader("fragment")]
Output fragmentMain()
{
    return { float4(1), {float4(2)}, float4(3) };
}
