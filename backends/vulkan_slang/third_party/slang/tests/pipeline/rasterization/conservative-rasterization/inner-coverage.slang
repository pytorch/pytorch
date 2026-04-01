// inner-coverage.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment
//TEST:SIMPLE(filecheck=CHECK):-emit-spirv-directly -target spirv -entry main -stage fragment

// CHECK: OpCapability FragmentFullyCoveredEXT
// CHECK: OpExtension "SPV_EXT_fragment_fully_covered"
// CHECK: OpEntryPoint Fragment %main "main"
// CHECK: OpDecorate %{{.*}} BuiltIn FullyCoveredEXT

[shader("fragment")]
void main(
    uint innerCoverage : SV_InnerCoverage,
    out float4 result : SV_Target)
{
    result = innerCoverage;
}
