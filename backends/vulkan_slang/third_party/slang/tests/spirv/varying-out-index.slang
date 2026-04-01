// Test that explicit binding of varying output works.

//TEST:SIMPLE(filecheck=CHECK): -stage fragment -entry MainPS -target spirv -emit-spirv-directly

struct PS_OUTPUT 
{
    [[vk::location(0) vk::index(0)]]    
    float4 vColor : SV_Target0 ; 
    
    [[vk::location(0) vk::index(1)]] 
    float4 vColor2 : SV_Target1 ; 
    
};

// CHECK: OpDecorate %{{.*}}MainPS_vColor Location 0
// CHECK: OpDecorate %{{.*}}MainPS_vColor2 Location 0
// CHECK: OpDecorate %{{.*}}MainPS_vColor2 Index 1

PS_OUTPUT MainPS()
{
    PS_OUTPUT output;
    output.vColor = float4(1.0f, 0.0f, 0.0f, 1.0f);
    output.vColor2 = float4(0.0f, 1.0f, 0.0f, 1.0f);
    return output;
}
