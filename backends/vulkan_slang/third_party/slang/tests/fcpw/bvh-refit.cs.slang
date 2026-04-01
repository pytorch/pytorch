import fcpw;

#define UNDEFINED_BVH_TYPE 0
#define LINE_SEGMENT_BVH 1
#define TRIANGLE_BVH 2
#define LINE_SEGMENT_SNCH 3
#define TRIANGLE_SNCH 4

#ifndef _BVH_TYPE
#define _BVH_TYPE UNDEFINED_BVH_TYPE
#endif

#if _BVH_TYPE == LINE_SEGMENT_BVH
uniform ParameterBlock<Bvh<BvhNode, LineSegment, NoSilhouette>> gBvh;

#elif _BVH_TYPE == TRIANGLE_BVH
uniform ParameterBlock<Bvh<BvhNode, Triangle, NoSilhouette>> gBvh;

#elif _BVH_TYPE == LINE_SEGMENT_SNCH
uniform ParameterBlock<Bvh<SnchNode, LineSegment, Vertex>> gBvh;

#elif _BVH_TYPE == TRIANGLE_SNCH
uniform ParameterBlock<Bvh<SnchNode, Triangle, Edge>> gBvh;

#else
// Compile time error
#error _BVH_TYPE is not set to a supported type
#endif

[shader("compute")]
[numthreads(256, 1, 1)]
void refit(uint3 threadId: SV_DispatchThreadID,
           uniform StructuredBuffer<uint> nodeIndices,
           uniform uint firstNodeOffset,
           uniform uint nodeCount)
{
    uint index = threadId.x;
    if (index >= nodeCount)
    {
        return;
    }

    uint nodeIndex = nodeIndices[firstNodeOffset + index];
    gBvh.refit(nodeIndex);
}
