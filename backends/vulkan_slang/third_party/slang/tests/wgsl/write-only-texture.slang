//TEST:REFLECTION(filecheck=REFLECT):-stage compute -entry computeMain -target wgsl -no-codegen
//TEST:SIMPLE(filecheck=SPV): -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=SPV): -target spirv -emit-spirv-via-glsl
//TEST:SIMPLE(filecheck=CHECK): -target wgsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK-WGSL-SPV): -target wgsl-spirv-asm -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=HLSL): -target hlsl -entry computeMain -profile cs_6_0
//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry computeMain -stage compute

// CHECK: texture_storage_2d<rgba8unorm, write>

// CHECK-WGSL-SPV: OpDecorate {{.*}} NonReadable
// CHECK-WGSL-SPV: OpTypeImage %float 2D {{.}} 0 0 2 Rgba8

// SPV: OpDecorate {{.*}} NonReadable
// SPV: OpTypeImage %float 2D {{.}} 0 0 2 Rgba8

// HLSL: RWTexture2D

// GLSL: writeonly
// GLSL: image2D

// REFLECT: "name": "writeOnlyTexture"
// REFLECT: "access": "write"

[format("rgba8")]
WTexture2D writeOnlyTexture;

[shader("compute")]
void computeMain()
{
    writeOnlyTexture.Store(int2(0,0), float4(0.2, 0.3, 0.4, 1.0));
}
