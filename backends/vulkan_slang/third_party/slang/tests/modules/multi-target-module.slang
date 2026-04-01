// multi-target-module.slang
 // Test that a slang-module can store both SPIR-V and DXIL blobs separately
 
 //TEST:SIMPLE(filecheck=CHECK): -o tests/modules/multi-target-module.slang-module -target dxil -embed-downstream-ir -target spirv -embed-downstream-ir -profile lib_6_6 -incomplete-library -dump-ir -verbose-paths
 
 module multi_target_module;
 
 // Simple function that will work on both SPIR-V and DXIL targets
 public float4 addVectors(float4 a, float4 b)
 {
     return a + b;
 }
 
 // Another function that should be compatible with both targets
 public float3 normalizeVector(float3 v)
 {
     return normalize(v);
 }
 
 [shader("compute")]
 [numthreads(8, 8, 1)]
 void main(uint3 dispatchThreadID : SV_DispatchThreadID)
 {
     float4 a = float4(1.0, 2.0, 3.0, 4.0);
     float4 b = float4(5.0, 6.0, 7.0, 8.0);
     
     float4 result = addVectors(a, b);
     
     float3 v = float3(1.0, 1.0, 1.0);
     float3 n = normalizeVector(v);
 } 
 
 // Check for the first occurrence of availableInDownstreamIR for addVectors in this section
 // Check that there are two entries, one for dxil and one for spirv.
 // CHECK: [availableInDownstreamIR(6 : Int)]
 // CHECK: [availableInDownstreamIR(10 : Int)]
 // CHECK: [public]
 // CHECK: [export("_S19multi_target_module10addVectorsp2pi_v4fi_v4fv4f")]

 // Check for the second occurrence of availableInDownstreamIR for normalizeVector in this section
 // Check that there are two entries, one for dxil and one for spirv.
 // CHECK: [availableInDownstreamIR(6 : Int)]
 // CHECK: [availableInDownstreamIR(10 : Int)]
 // CHECK: [public]
 // CHECK: [export("_S19multi_target_module15normalizeVectorp1pi_v3fv3f")]