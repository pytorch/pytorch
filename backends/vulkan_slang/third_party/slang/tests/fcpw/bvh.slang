implementing fcpw;
__include aggregate;
__include geometry;
__include bvh_node;

public static const uint FCPW_BVH_MAX_DEPTH = 64;

public struct TraversalStack
{
    public uint node;      // node index
    public float distance; // minimum distance (parametric, squared, ...) to this node

    // constructor
    public __init()
    {
        node = 0;
        distance = 0.0;
    }
};

public struct Bvh<N : IBvhNode, P : IPrimitive, S : ISilhouette> : IAggregate
{
    public RWStructuredBuffer<N> nodes;
    public StructuredBuffer<P> primitives;
    public StructuredBuffer<S> silhouettes;

    // updates the bounding volume of an aggregate leaf node
    [mutating]
    internal void refitLeafNode(uint nodeIndex)
    {
        // update leaf node's bounding box
        float3 pMin = float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 pMax = float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        N node = nodes[nodeIndex];
        uint nPrimitives = node.getNumPrimitives();

        for (uint p = 0; p < nPrimitives; p++)
        {
            uint primitiveIndex = node.getPrimitiveOffset() + p;
            BoundingBox primitiveBox = primitives[primitiveIndex].getBoundingBox();
            pMin = min(pMin, primitiveBox.pMin);
            pMax = max(pMax, primitiveBox.pMax);
        }

        node.setBoundingBox(BoundingBox(pMin, pMax));

        if (node.hasBoundingCone())
        {
            // update leaf node's bounding cone
            float3 axis = float3(0.0, 0.0, 0.0);
            float3 centroid = 0.5 * (pMin + pMax);
            float halfAngle = 0.0;
            float radius = 0.0;
            bool anySilhouettes = false;
            bool silhouettesHaveTwoAdjacentFaces = true;
            uint nSilhouettes = node.getNumSilhouettes();

            for (uint p = 0; p < nSilhouettes; p++)
            {
                uint silhouetteIndex = node.getSilhouetteOffset() + p;
                S silhouette = silhouettes[silhouetteIndex];
                axis += silhouette.getNormal(0);
                axis += silhouette.getNormal(1);
                radius = max(radius, length(silhouette.getCentroid() - centroid));
                silhouettesHaveTwoAdjacentFaces = silhouettesHaveTwoAdjacentFaces &&
                                                  silhouette.hasTwoAdjacentFaces();
                anySilhouettes = true;
            }

            if (!anySilhouettes)
            {
                halfAngle = -M_PI;
            }
            else if (!silhouettesHaveTwoAdjacentFaces)
            {
                halfAngle = M_PI;
            }
            else
            {
                float axisNorm = length(axis);
                if (axisNorm > FLT_EPSILON)
                {
                    axis /= axisNorm;

                    for (uint p = 0; p < nSilhouettes; p++)
                    {
                        uint silhouetteIndex = node.getSilhouetteOffset() + p;
                        for (uint k = 0; k < 2; k++)
                        {
                            float3 n = silhouettes[silhouetteIndex].getNormal(k);
                            float angle = acos(max(-1.0, min(1.0, dot(axis, n))));
                            halfAngle = max(halfAngle, angle);
                        }
                    }
                }
            }

            node.setBoundingCone(BoundingCone(axis, halfAngle, radius));
        }
    }

    // updates the bounding volume of an aggregate internal node
    [mutating]
    internal void refitInternalNode(uint nodeIndex)
    {
        // update internal node's bounding box
        N node = nodes[nodeIndex];
        uint leftNodeIndex = nodeIndex + 1;
        uint rightNodeIndex = nodeIndex + node.getRightChildOffset();
        N leftNode = nodes[leftNodeIndex];
        N rightNode = nodes[rightNodeIndex];

        BoundingBox leftBox = leftNode.getBoundingBox();
        BoundingBox rightBox = rightNode.getBoundingBox();
        BoundingBox mergedBox = mergeBoundingBoxes(leftBox, rightBox);
        node.setBoundingBox(mergedBox);

        if (node.hasBoundingCone())
        {
            // update internal node's bounding cone
            BoundingCone leftCone = leftNode.getBoundingCone();
            BoundingCone rightCone = rightNode.getBoundingCone();
            BoundingCone mergedCone = mergeBoundingCones(leftCone, rightCone,
                                                         leftBox.getCentroid(),
                                                         rightBox.getCentroid(),
                                                         mergedBox.getCentroid());
            node.setBoundingCone(mergedCone);
        }
    }

    // updates the bounding volume of an aggregate node
    // NOTE: assumes node indices are provided in bottom-up order
    [mutating]
    public void refit(uint nodeIndex)
    {
        if (nodes[nodeIndex].isLeaf())
        {
            refitLeafNode(nodeIndex);
        }
        else
        {
            refitInternalNode(nodeIndex);
        }
    }

    // intersects aggregate geometry with ray
    public bool intersect(inout Ray r, bool checkForOcclusion, inout Interaction i)
    {
        TraversalStack traversalStack[FCPW_BVH_MAX_DEPTH];
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        bool didIntersect = false;

        if (rootBox.intersect(r, distToChildNodes[0], distToChildNodes[1]))
        {
            traversalStack[0].node = 0;
            traversalStack[0].distance = distToChildNodes[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                uint currentNodeIndex = traversalStack[stackPtr].node;
                float currentDist = traversalStack[stackPtr].distance;
                stackPtr--;

                // if this node is further than the closest found intersection, continue
                if (currentDist > r.tMax)
                {
                    continue;
                }

                N node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    // intersect primitives in leaf node
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        Interaction c;
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        bool didIntersectPrimitive = primitives[primitiveIndex].intersect(r, checkForOcclusion, c);

                        if (didIntersectPrimitive)
                        {
                            if (checkForOcclusion)
                            {
                                i.index = c.index;
                                return true;
                            }

                            didIntersect = true;
                            r.tMax = min(r.tMax, c.d);
                            i = c;
                        }
                    }
                }
                else
                {
                    // intersect child nodes
                    uint leftNodeIndex = currentNodeIndex + 1;
                    BoundingBox leftBox = nodes[leftNodeIndex].getBoundingBox();
                    bool didIntersectLeft = leftBox.intersect(r, distToChildNodes[0], distToChildNodes[1]);

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    BoundingBox rightBox = nodes[rightNodeIndex].getBoundingBox();
                    bool didIntersectRight = rightBox.intersect(r, distToChildNodes[2], distToChildNodes[3]);

                    // which nodes did we intersect?
                    if (didIntersectLeft && didIntersectRight)
                    {
                        // assume that the left child is closer
                        uint closer = leftNodeIndex;
                        uint other = rightNodeIndex;

                        // ... if the right child was actually closer, swap the relavent values
                        if (distToChildNodes[2] < distToChildNodes[0])
                        {
                            float tmpDist = distToChildNodes[0];
                            distToChildNodes[0] = distToChildNodes[2];
                            distToChildNodes[2] = tmpDist;

                            uint tmpNodeIndex = closer;
                            closer = other;
                            other = tmpNodeIndex;
                        }

                        // it's possible that the nearest primitive is still in the other node,
                        // but we'll check the farther-away node later.

                        // push the further node first, then the closer node
                        stackPtr++;
                        traversalStack[stackPtr].node = other;
                        traversalStack[stackPtr].distance = distToChildNodes[2];

                        stackPtr++;
                        traversalStack[stackPtr].node = closer;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (didIntersectLeft)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = leftNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (didIntersectRight)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = rightNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[2];
                    }
                }
            }
        }

        return didIntersect;
    }

    // intersects aggregate geometry with sphere
    public bool intersect<T : IBranchTraversalWeight>(BoundingSphere s, float3 randNums,
                                                      T branchTraversalWeight,
                                                      inout Interaction i)
    {
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        uint currentNodeIndex = 0;
        uint selectedPrimitiveIndex = UINT_MAX;
        bool didIntersect = false;

        if (rootBox.overlap(s, distToChildNodes[0], distToChildNodes[1]))
        {
            float maxDistToChildNode = distToChildNodes[1];
            float traversalPdf = 1.0;
            float u = randNums[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                stackPtr--;

                N node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    // probabilistically select a primitive
                    float totalPrimitiveWeight = 0.0;
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        Interaction c;
                        bool didIntersectPrimitive = false;
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        P primitive = primitives[primitiveIndex];

                        if (maxDistToChildNode <= s.r2)
                        {
                            didIntersectPrimitive = true;
                            c.d = primitive.getSurfaceArea();
                            c.index = primitive.getIndex();
                        }
                        else
                        {
                            didIntersectPrimitive = primitive.intersect(s, c);
                        }

                        if (didIntersectPrimitive)
                        {
                            didIntersect = true;
                            totalPrimitiveWeight += c.d;
                            float selectionProb = c.d / totalPrimitiveWeight;

                            if (u < selectionProb)
                            {
                                u = u / selectionProb; // rescale to [0,1)
                                i = c;
                                i.d *= traversalPdf;
                                selectedPrimitiveIndex = primitiveIndex;
                            }
                            else
                            {
                                u = (u - selectionProb) / (1.0 - selectionProb);
                            }
                        }
                    }

                    if (totalPrimitiveWeight > 0.0)
                    {
                        i.d /= totalPrimitiveWeight;
                    }
                }
                else
                {
                    // probabilistically select one child node to traverse
                    uint leftNodeIndex = currentNodeIndex + 1;
                    BoundingBox leftBox = nodes[leftNodeIndex].getBoundingBox();
                    bool overlapsLeft = leftBox.overlap(s, distToChildNodes[0], distToChildNodes[1]);
                    float weightLeft = overlapsLeft ? 1.0 : 0.0;
                    if (weightLeft > 0.0)
                    {
                        float3 u = s.c - leftBox.getCentroid();
                        weightLeft *= branchTraversalWeight.compute(dot(u, u));
                    }

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    BoundingBox rightBox = nodes[rightNodeIndex].getBoundingBox();
                    bool overlapsRight = rightBox.overlap(s, distToChildNodes[2], distToChildNodes[3]);
                    float weightRight = overlapsRight ? 1.0 : 0.0;
                    if (weightRight > 0.0)
                    {
                        float3 u = s.c - rightBox.getCentroid();
                        weightRight *= branchTraversalWeight.compute(dot(u, u));
                    }

                    float totalTraversalWeight = weightLeft + weightRight;
                    if (totalTraversalWeight > 0.0)
                    {
                        stackPtr++;
                        float traversalProbLeft = weightLeft / totalTraversalWeight;
                        float traversalProbRight = 1.0 - traversalProbLeft;

                        if (u < traversalProbLeft)
                        {
                            u = u / traversalProbLeft; // rescale to [0,1)
                            currentNodeIndex = leftNodeIndex;
                            traversalPdf *= traversalProbLeft;
                            maxDistToChildNode = distToChildNodes[1];
                        }
                        else
                        {
                            u = (u - traversalProbLeft) / traversalProbRight; // rescale to [0,1)
                            currentNodeIndex = rightNodeIndex;
                            traversalPdf *= traversalProbRight;
                            maxDistToChildNode = distToChildNodes[3];
                        }
                    }
                }
            }
        }

        if (didIntersect)
        {
            if (i.index == UINT_MAX || selectedPrimitiveIndex == UINT_MAX)
            {
                didIntersect = false;
            }
            else
            {
                // sample a point on the selected geometric primitive
                float samplingPdf = primitives[selectedPrimitiveIndex].samplePoint(randNums.yz, i.uv, i.p, i.n);
                i.d *= samplingPdf;
            }
        }

        return didIntersect;
    }

    // finds closest point on aggregate geometry from sphere center
    public bool findClosestPoint(inout BoundingSphere s, inout Interaction i,
                                 bool recordNormal = false)
    {
        TraversalStack traversalStack[FCPW_BVH_MAX_DEPTH];
        float4 distToChildNodes = float4(0.0, 0.0, 0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        bool notFound = true;

        if (rootBox.overlap(s, distToChildNodes[0], distToChildNodes[1]))
        {
            s.r2 = min(s.r2, distToChildNodes[1]);
            traversalStack[0].node = 0;
            traversalStack[0].distance = distToChildNodes[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                uint currentNodeIndex = traversalStack[stackPtr].node;
                float currentDist = traversalStack[stackPtr].distance;
                stackPtr--;

                // if this node is further than the closest found primitive, continue
                if (currentDist > s.r2)
                {
                    continue;
                }

                N node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    // compute distance to primitives in leaf node
                    uint nPrimitives = node.getNumPrimitives();
                    for (uint p = 0; p < nPrimitives; p++)
                    {
                        Interaction c;
                        uint primitiveIndex = node.getPrimitiveOffset() + p;
                        bool found = primitives[primitiveIndex].findClosestPoint(s, c);

                        // keep the closest point only
                        if (found)
                        {
                            notFound = false;
                            s.r2 = min(s.r2, c.d * c.d);
                            i = c;
                        }
                    }
                }
                else
                {
                    // find distance to child nodes
                    uint leftNodeIndex = currentNodeIndex + 1;
                    BoundingBox leftBox = nodes[leftNodeIndex].getBoundingBox();
                    bool overlapsLeft = leftBox.overlap(s, distToChildNodes[0], distToChildNodes[1]);
                    s.r2 = min(s.r2, distToChildNodes[1]);

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    BoundingBox rightBox = nodes[rightNodeIndex].getBoundingBox();
                    bool overlapsRight = rightBox.overlap(s, distToChildNodes[2], distToChildNodes[3]);
                    s.r2 = min(s.r2, distToChildNodes[3]);

                    // which nodes do we overlap?
                    if (overlapsLeft && overlapsRight)
                    {
                        // assume that the left child is closer
                        uint closer = leftNodeIndex;
                        uint other = rightNodeIndex;

                        // ... if the right child was actually closer, swap the relavent values
                        if (distToChildNodes[0] == 0.0 && distToChildNodes[2] == 0.0)
                        {
                            if (distToChildNodes[3] < distToChildNodes[1])
                            {
                                uint tmpNodeIndex = closer;
                                closer = other;
                                other = tmpNodeIndex;
                            }
                        }
                        else if (distToChildNodes[2] < distToChildNodes[0])
                        {
                            float tmpDist = distToChildNodes[0];
                            distToChildNodes[0] = distToChildNodes[2];
                            distToChildNodes[2] = tmpDist;

                            uint tmpNodeIndex = closer;
                            closer = other;
                            other = tmpNodeIndex;
                        }

                        // it's possible that the nearest primitive is still in the other node,
                        // but we'll check the farther-away node later.

                        // push the further node first, then the closer node
                        stackPtr++;
                        traversalStack[stackPtr].node = other;
                        traversalStack[stackPtr].distance = distToChildNodes[2];

                        stackPtr++;
                        traversalStack[stackPtr].node = closer;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (overlapsLeft)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = leftNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (overlapsRight)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = rightNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[2];
                    }
                }
            }
        }

        if (!notFound && recordNormal)
        {
            i.n = primitives[i.index].getNormal();
        }

        return !notFound;
    }

    // finds closest silhouette point on aggregate geometry from sphere center
    public bool findClosestSilhouettePoint(inout BoundingSphere s, bool flipNormalOrientation,
                                           float squaredMinRadius, float precision,
                                           inout Interaction i)
    {
        if (squaredMinRadius >= s.r2)
        {
            return false;
        }

        TraversalStack traversalStack[FCPW_BVH_MAX_DEPTH];
        float2 distToChildNodes = float2(0.0, 0.0);
        BoundingBox rootBox = nodes[0].getBoundingBox();
        bool notFound = true;

        if (rootBox.overlap(s, distToChildNodes[0]))
        {
            traversalStack[0].node = 0;
            traversalStack[0].distance = distToChildNodes[0];
            int stackPtr = 0;

            while (stackPtr >= 0)
            {
                // pop off the next node to work on
                uint currentNodeIndex = traversalStack[stackPtr].node;
                float currentDist = traversalStack[stackPtr].distance;
                stackPtr--;

                // if this node is further than the closest found primitive, continue
                if (currentDist > s.r2)
                {
                    continue;
                }

                N node = nodes[currentNodeIndex];
                if (node.isLeaf())
                {
                    // compute distance to silhouettes in leaf node
                    uint nSilhouettes = node.getNumSilhouettes();
                    for (uint p = 0; p < nSilhouettes; p++)
                    {
                        uint silhouetteIndex = node.getSilhouetteOffset() + p;
                        S silhouette = silhouettes[silhouetteIndex];
                        if (silhouette.getIndex() == i.index)
                        {
                            // silhouette has already been checked
                            continue;
                        }

                        Interaction c;
                        bool found = silhouette.findClosestSilhouettePoint(
                            s, flipNormalOrientation, squaredMinRadius, precision, c);

                        // keep the closest silhouette point
                        if (found)
                        {
                            notFound = false;
                            s.r2 = min(s.r2, c.d * c.d);
                            i = c;

                            if (squaredMinRadius >= s.r2)
                            {
                                break;
                            }
                        }
                    }
                }
                else
                {
                    // find distance to child nodes
                    // NOTE: Slang does not support short-circuiting with the && operator, hence the clunky code
                    uint leftNodeIndex = currentNodeIndex + 1;
                    N leftNode = nodes[leftNodeIndex];
                    BoundingCone leftCone = leftNode.getBoundingCone();
                    bool overlapsLeft = leftCone.isValid();
                    if (overlapsLeft)
                    {
                        BoundingBox leftBox = leftNode.getBoundingBox();
                        overlapsLeft = leftBox.overlap(s, distToChildNodes[0]);
                        if (overlapsLeft)
                        {
                            float minAngleRange, maxAngleRange;
                            overlapsLeft = leftCone.overlap(s.c, leftBox, distToChildNodes[0],
                                                            minAngleRange, maxAngleRange);
                        }
                    }

                    uint rightNodeIndex = currentNodeIndex + node.getRightChildOffset();
                    N rightNode = nodes[rightNodeIndex];
                    BoundingCone rightCone = rightNode.getBoundingCone();
                    bool overlapsRight = rightCone.isValid();
                    if (overlapsRight)
                    {
                        BoundingBox rightBox = rightNode.getBoundingBox();
                        overlapsRight = rightBox.overlap(s, distToChildNodes[1]);
                        if (overlapsRight)
                        {
                            float minAngleRange, maxAngleRange;
                            overlapsRight = rightCone.overlap(s.c, rightBox, distToChildNodes[1],
                                                              minAngleRange, maxAngleRange);
                        }
                    }

                    // which nodes do we overlap?
                    if (overlapsLeft && overlapsRight)
                    {
                        // assume that the left child is closer
                        uint closer = leftNodeIndex;
                        uint other = rightNodeIndex;

                        // ... if the right child was actually closer, swap the relavent values
                        if (distToChildNodes[1] < distToChildNodes[0])
                        {
                            float tmpDist = distToChildNodes[0];
                            distToChildNodes[0] = distToChildNodes[1];
                            distToChildNodes[1] = tmpDist;

                            uint tmpNodeIndex = closer;
                            closer = other;
                            other = tmpNodeIndex;
                        }

                        // it's possible that the nearest primitive is still in the other node,
                        // but we'll check the farther-away node later.

                        // push the further node first, then the closer node
                        stackPtr++;
                        traversalStack[stackPtr].node = other;
                        traversalStack[stackPtr].distance = distToChildNodes[1];

                        stackPtr++;
                        traversalStack[stackPtr].node = closer;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (overlapsLeft)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = leftNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[0];
                    }
                    else if (overlapsRight)
                    {
                        stackPtr++;
                        traversalStack[stackPtr].node = rightNodeIndex;
                        traversalStack[stackPtr].distance = distToChildNodes[1];
                    }
                }
            }
        }

        return !notFound;
    }
};
