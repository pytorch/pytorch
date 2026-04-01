//TEST:SIMPLE(filecheck=CHECK): -target spirv -profile glsl_450 -stage compute -entry MainCs -line-directive-mode none
//TEST:SIMPLE(filecheck=CHECK): -target spirv -profile glsl_450 -stage compute -entry MainCs -line-directive-mode none -minimum-slang-optimization

//CHECK: {{.*}} OpAtomicIAdd
RWBuffer<uint> g_InterlockTest;

[numthreads(1,1,1)]
void MainCs( uint2 dispatchThreadID : SV_DispatchThreadID, uint2 groupThreadID : SV_GroupThreadID )
{
	uint nVertexCount = 41;
	uint nVertexBufferSize = 40;
	uint nVertex = 3;

	InterlockedAdd( g_InterlockTest[ 0 ], nVertexCount, nVertex );

	// Did we overflow?
	if ( nVertex + nVertexCount > nVertexBufferSize )
	{
		InterlockedAdd( g_InterlockTest[ 0 ], -int(nVertexCount) );
	}
}
