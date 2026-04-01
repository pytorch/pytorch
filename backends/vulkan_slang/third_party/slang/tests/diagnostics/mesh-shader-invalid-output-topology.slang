//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_HLSL): -entry main1 -stage mesh -target hlsl
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -entry main1 -stage mesh -target spirv
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -entry main1 -stage mesh -target glsl
//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -entry main1 -stage mesh -target metal

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK_POINT_HLSL): -entry main2 -stage mesh -target hlsl
//TEST:SIMPLE(filecheck=CHECK_POINT): -entry main2 -stage mesh -target spirv
//TEST:SIMPLE(filecheck=CHECK_POINT): -entry main2 -stage mesh -target glsl
//TEST:SIMPLE(filecheck=CHECK_POINT): -entry main2 -stage mesh -target metal


struct TaskPayload {
    uint offset;
};

struct Output {
    float4 position : SV_POSITION;
};


// CHECK_HLSL: 50060: Invalid{{.*}}asdqwe{{.*}}of: 'line', 'triangle'
// CHECK: 50060: Invalid{{.*}}asdqwe{{.*}}of: 'point', 'line', 'triangle'
[numthreads(32, 1, 1)]
[outputtopology("asdqwe")]
[shader("mesh")]
void main1(
    uint ThreadIndex: SV_GroupIndex,
    uint GroupID: SV_GroupID,
    out vertices Output Vertices[64],
    out indices uint3 Triangles[124],
    in payload TaskPayload Payload
) {
}

// 'point' is not valid for HLSL only, other targets must compile successfully.
// CHECK_POINT_HLSL: 50060: Invalid{{.*}}point{{.*}}of: 'line', 'triangle'
// CHECK_POINT: main
[numthreads(32, 1, 1)]
[outputtopology("point")]
[shader("mesh")]
void main2(
    uint ThreadIndex: SV_GroupIndex,
    uint GroupID: SV_GroupID,
    out vertices Output Vertices[64],
    out indices uint Points[124],
    in payload TaskPayload Payload
) {
}
