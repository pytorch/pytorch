implementing fcpw;
__include ray;
__include interaction;
__include bounding_volumes;
__include transform;

public interface IBranchTraversalWeight
{
    // computes the traversal weight for a given squared distance
    float compute(float r2);
};

public struct ConstantBranchTraversalWeight : IBranchTraversalWeight
{
    // computes the traversal weight for a given squared distance
    public float compute(float r2)
    {
        return 1.0;
    }
};

public interface IAggregate
{
    // updates the bounding volume of an aggregate node
    [mutating]
    void refit(uint nodeIndex);

    // intersects aggregate geometry with ray
    bool intersect(inout Ray r, bool checkForOcclusion, inout Interaction i);

    // intersects aggregate geometry with sphere
    bool intersect<T : IBranchTraversalWeight>(BoundingSphere s, float3 randNums,
                                               T branchTraversalWeight,
                                               inout Interaction i);

    // finds closest point on aggregate geometry from sphere center
    bool findClosestPoint(inout BoundingSphere s, inout Interaction i,
                          bool recordNormal = false);

    // finds closest silhouette point on aggregate geometry from sphere center
    bool findClosestSilhouettePoint(inout BoundingSphere s, bool flipNormalOrientation,
                                    float squaredMinRadius, float precision,
                                    inout Interaction i);
};

public struct TransformedAggregate<A : IAggregate> : IAggregate
{
    public A aggregate;
    public float3x4 t;
    public float3x4 tInv;

    // updates the bounding volume of an aggregate node
    // NOTE: refitting of transformed aggregates is currently quite inefficient, since the
    // shared aggregate is refit every time this function is called
    [mutating]
    public void refit(uint nodeIndex)
    {
        aggregate.refit(nodeIndex);
    }

    // intersects aggregate geometry with ray
    public bool intersect(inout Ray r, bool checkForOcclusion, inout Interaction i)
    {
        // apply inverse transform to ray
        Ray rInv = transformRay(tInv, r);

        // intersect
        bool didIntersect = aggregate.intersect(rInv, checkForOcclusion, i);

        // apply transform to ray and interaction
        r.tMax = transformRay(t, rInv).tMax;
        if (didIntersect)
        {
            transformInteraction(t, tInv, r.o, true, i);
            return true;
        }

        return false;
    }

    // intersects aggregate geometry with sphere
    public bool intersect<T : IBranchTraversalWeight>(BoundingSphere s, float3 randNums,
                                                      T branchTraversalWeight,
                                                      inout Interaction i)
    {
        // apply inverse transform to sphere
        BoundingSphere sInv = transformSphere(tInv, s);

        // intersect
        bool didIntersect = aggregate.intersect(sInv, randNums, branchTraversalWeight, i);

        // apply transform to interaction
        if (didIntersect)
        {
            transformInteraction(t, tInv, s.c, false, i);
            return true;
        }

        return false;
    }

    // finds closest point on aggregate geometry from sphere center
    public bool findClosestPoint(inout BoundingSphere s, inout Interaction i,
                                 bool recordNormal = false)
    {
        // apply inverse transform to sphere
        BoundingSphere sInv = transformSphere(tInv, s);

        // find closest point
        bool didFindClosestPoint = aggregate.findClosestPoint(sInv, i, recordNormal);

        // apply transform to sphere and interaction
        s.r2 = transformSphere(t, sInv).r2;
        if (didFindClosestPoint)
        {
            transformInteraction(t, tInv, s.c, true, i);
            return true;
        }

        return false;
    }

    // finds closest silhouette point on aggregate geometry from sphere center
    public bool findClosestSilhouettePoint(inout BoundingSphere s, bool flipNormalOrientation,
                                           float squaredMinRadius, float precision,
                                           inout Interaction i)
    {
        // apply inverse transform to sphere
        BoundingSphere sInv = transformSphere(tInv, s);
        BoundingSphere sMin = BoundingSphere(s.c, squaredMinRadius);
        BoundingSphere sMinInv = transformSphere(tInv, sMin);

        // find closest silhouette point
        bool didFindClosestSilhouettePoint = aggregate.findClosestSilhouettePoint(
            sInv, flipNormalOrientation, sMinInv.r2, precision, i);

        // apply transform to sphere and interaction
        s.r2 = transformSphere(t, sInv).r2;
        if (didFindClosestSilhouettePoint)
        {
            transformInteraction(t, tInv, s.c, true, i);
            return true;
        }

        return false;
    }
};
