// sign.slang

//TEST:SIMPLE(filecheck=SPIRV):-target spirv-assembly -entry main -stage fragment
//TEST:SIMPLE(filecheck=DXIL):-target dxil-assembly  -entry main -stage fragment -profile sm_6_0

// Test cross compilation of the sign function

// SPIRV: FSign
// DXIL: define void @main

float4 main() : SV_Target
{
    float4 s = sign(float4(1.5, 1.0, -1.5, -1.0));
    return s;
}

