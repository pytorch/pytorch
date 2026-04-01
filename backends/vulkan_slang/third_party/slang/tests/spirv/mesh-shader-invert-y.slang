//TEST:SIMPLE(filecheck=CHECK): -target spirv -fvk-invert-y

// CHECK: %[[CONST:[0-9]+]] = OpConstantComposite %v4float %float_0 %float_n1 %float_0 %float_0
// CHECK: OpStore {{.*}} %[[CONST]]

struct PS_IN
{
	float4 vPositionPs : SV_Position;
};

[ shader("mesh") ]
[ outputtopology( "line" ) ]
[ numthreads( 64, 1, 1 ) ]
void main(
	uint nThreadId : SV_DispatchThreadID,
	uint nGroupThreadId : SV_GroupThreadID,
	uint nGroupId : SV_GroupID,
	out vertices PS_IN outputVerts[ 128 ],
	out indices uint2 outputIB[ 64  ] )
{
	SetMeshOutputCounts( 128, 64 );

    outputVerts[ 2 * nGroupThreadId ].vPositionPs = float4( 0, 1, 0, 0 );
}
