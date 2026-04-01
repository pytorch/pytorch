// trace-ray-inline.slang

//TEST:CROSS_COMPILE:-target dxil-asm -stage compute -profile sm_6_5 -entry main -line-directive-mode none
//TEST:SIMPLE(filecheck=CHECK):-target spirv-asm -stage compute -profile glsl_460+GL_EXT_ray_query -entry main -line-directive-mode none

// CHECK: OpCapability RayQueryKHR
// CHECK: OpExtension "SPV_KHR_ray_query"
// CHECK: OpRayQueryInitializeKHR
// CHECK: OpRayQueryProceedKHR
// CHECK: OpRayQueryGetIntersectionTypeKHR
// CHECK: OpRayQueryConfirmIntersectionKHR

// The goal of this shader is to use all the main pieces
// of functionality in DXR 1.1's `TraceRayInline` feature,
// to ensure that they survive translation to HLSL.

// In order to trace rays, we need an acceleration structure.
//
RaytracingAccelerationStructure myAccelerationStructure;

// We also need to decide what to do with hits/misses.
// The `TraceRayInline` approach eschews separate shader
// stages for RT, and instead expects users to write
// those operations as subroutines instead.
//
// We will mimic the style and naming of DXR 1.0 here
// to try and make the parallels clear.
//
// We start with a ray "payload" type that will be
// used for input/output on hit and miss shaders
//

struct MyRayPayload
{
	int value;
};

// The first and simplest shader is the miss shader.
//
void myMiss(inout MyRayPayload payload)
{
	payload.value = 0;
}

// Next, up is a closest hit shader for opaque triangles.
//
void myTriangleClosestHit(inout MyRayPayload payload)
{
	payload.value = 1;
}

// In order to support alpha testing, we need an any-hit
// shader for triangles.
//
// In this case, the return value is used to specify
// whether the hit should be accepted (true) or ignored (false).
//
bool myTriangleAnyHit(inout MyRayPayload payload)
{
    unmodified(payload);
	return true;
}

// Procedural primitives are different than triangles
// in that they need user-defined hit attributes.
//
struct MyProceduralHitAttrs { int value; }

// Otherwise, the closest- and any-hit shaders
// for procedural primitives are similar to those
// for triangles.
//
void myProceduralClosestHit(inout MyRayPayload payload, MyProceduralHitAttrs attrs)
{
	payload.value = attrs.value;
}
bool myProceduralAnyHit(inout MyRayPayload payload)
{
    unmodified(payload);
	return true;
}

// The new piece of the puzzle for procedural primitives
// is the intersection shader, which should be able to
// report zero or more intersections.
//
// For now we will only deal with the single-intersection
// case.
//
bool myProceduralIntersection(inout float tHit, inout MyProceduralHitAttrs hitAttrs)
{
    unmodified(tHit);
    unmodified(hitAttrs);
	return true;
}

RWStructuredBuffer<int> resultBuffer;

// In order to kick of tracing we need the properties of a ray
// query to trace, so we will pipe those in via a constant buffer.
//
cbuffer C
{
	float3 origin;
	float tMin;
	float3 direction;
	float tMax;
	uint rayFlags;
	uint instanceMask;
	uint shouldStopAtFirstHit;
}

// The actual tracing is handled by a compute shader,
// which here takes on the role of a ray generation shader.
//
void main(uint3 tid : SV_DispatchThreadID)
{
	uint index = tid.x;

	RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> query;
	MyProceduralHitAttrs committedProceduralAttrs;

	MyRayPayload payload = { -1 };
	RayDesc ray = { origin, tMin, direction, tMax };
	query.TraceRayInline(
		myAccelerationStructure,
		rayFlags,
		instanceMask,
		ray);


	for(;;)
	{
		if(!query.Proceed()) break;

		switch(query.CandidateType())
		{
		case CANDIDATE_PROCEDURAL_PRIMITIVE:
			{
				MyProceduralHitAttrs candidateProceduralAttrs = { 0 };
				float tHit = 0.0f;
				if(myProceduralIntersection(tHit, candidateProceduralAttrs))
				{
					if(myProceduralAnyHit(payload))
					{
						query.CommitProceduralPrimitiveHit(tHit);
						committedProceduralAttrs = candidateProceduralAttrs;
						if(shouldStopAtFirstHit != 0)
							query.Abort();
					}
				}
			}
			break;

		case CANDIDATE_NON_OPAQUE_TRIANGLE:
			{
				if(myTriangleAnyHit(payload))
				{
					query.CommitNonOpaqueTriangleHit();
					if(shouldStopAtFirstHit != 0)
						query.Abort();
				}
			}
			break;

		}


	}

	switch(query.CommittedStatus())
	{
	case COMMITTED_TRIANGLE_HIT:
		myTriangleClosestHit(payload);
		break;

	case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
		myProceduralClosestHit(payload, committedProceduralAttrs);
		break;

	case COMMITTED_NOTHING:
		myMiss(payload);
		break;
	}

    resultBuffer[index] = payload.value;
}
