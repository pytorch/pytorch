// array-of-buffers.slang

//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage fragment
//TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly  -entry main -stage fragment -profile sm_6_0

// SPIRV: OpEntryPoint
// DXIL: define void @main()

// This test ensures that we cross-compile arrays of structured/constant
// buffers into appropriate GLSL, where these are not first-class types.
//
// Note that this test does *not* currently test the case of passing
// a structured or constant buffer into a subroutine, which requires
// further work.

struct S { float4 f; };

cbuffer C
{
    uint index;
}

ConstantBuffer<S>           cb [3];
StructuredBuffer<S>         sb1[4];
RWStructuredBuffer<float4>  sb2[5];
ByteAddressBuffer           bb [6];

float4 main() : SV_Target
{
    return cb [index]       .f
         + sb1[index][index].f
         + sb2[index][index]
         + float4(bb[index].Load(int(index*4)));
}

