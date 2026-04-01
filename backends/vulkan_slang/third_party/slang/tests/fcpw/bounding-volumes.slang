implementing fcpw;
__include ray;
__include math_constants;

public struct BoundingSphere
{
    public float3 c; // sphere center
    public float r2; // sphere squared radius

    // constructor
    public __init(float3 c_, float r2_)
    {
        c = c_;
        r2 = r2_;
    }
};

public struct BoundingBox
{
    public float3 pMin; // aabb min position
    public float3 pMax; // aabb max position

    // constructor
    public __init(float3 pMin_, float3 pMax_)
    {
        pMin = pMin_;
        pMax = pMax_;
    }

    // checks whether box is valid
    public bool isValid()
    {
        return all(pMin <= pMax);
    }

    // returns box centroid
    public float3 getCentroid()
    {
        return 0.5 * (pMin + pMax);
    }

    // checks for overlap with sphere
    public bool overlap(BoundingSphere s, out float d2Min, out float d2Max)
    {
        float3 u = pMin - s.c;
        float3 v = s.c - pMax;
        float3 a = max(max(u, v), float3(0.0, 0.0, 0.0));
        float3 b = min(u, v);
        d2Min = dot(a, a);
        d2Max = dot(b, b);

        return d2Min <= s.r2;
    }

    // checks for overlap with sphere
    public bool overlap(BoundingSphere s, out float d2Min)
    {
        float3 u = pMin - s.c;
        float3 v = s.c - pMax;
        float3 a = max(max(u, v), float3(0.0, 0.0, 0.0));
        d2Min = dot(a, a);

        return d2Min <= s.r2;
    }

    // intersects box with ray
    public bool intersect(Ray r, out float tMin, out float tMax)
    {
        // slab test for ray box intersection
        // source: http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf
        float3 t0 = (pMin - r.o) * r.dInv;
        float3 t1 = (pMax - r.o) * r.dInv;
        float3 tNear = min(t0, t1);
        float3 tFar = max(t0, t1);

        float tNearMax = max(0.0, max(tNear.x, max(tNear.y, tNear.z)));
        float tFarMin = min(r.tMax, min(tFar.x, min(tFar.y, tFar.z)));
        if (tNearMax > tFarMin)
        {
            tMin = FLT_MAX;
            tMax = FLT_MAX;
            return false;
        }

        tMin = tNearMax;
        tMax = tFarMin;
        return true;
    }
};

public BoundingBox mergeBoundingBoxes(BoundingBox boxA, BoundingBox boxB)
{
    BoundingBox box;
    box.pMin = min(boxA.pMin, boxB.pMin);
    box.pMax = max(boxA.pMax, boxB.pMax);

    return box;
}

public bool inRange(float val, float low, float high)
{
    return val >= low && val <= high;
}

public void computeOrthonormalBasis(float3 n, out float3 b1, out float3 b2)
{
    // source: https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float sign = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;

    b1 = float3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}

public float projectToPlane(float3 n, float3 e)
{
    // compute orthonormal basis
    float3 b1, b2;
    computeOrthonormalBasis(n, b1, b2);

    // compute maximal projection radius
    float r1 = dot(e, abs(b1));
    float r2 = dot(e, abs(b2));
    return sqrt(r1 * r1 + r2 * r2);
}

public struct BoundingCone
{
    public float3 axis;     // cone axis
    public float halfAngle; // cone half angle
    public float radius;    // cone radius

    // constructors
    public __init()
    {
        axis = float3(0.0, 0.0, 0.0);
        halfAngle = M_PI;
        radius = 0.0;
    }
    public __init(float3 axis_, float halfAngle_, float radius_)
    {
        axis = axis_;
        halfAngle = halfAngle_;
        radius = radius_;
    }

    // checks whether cone is valid
    public bool isValid()
    {
        return halfAngle >= 0.0;
    }

    // check for overlap between this cone and the "view" cone defined by the given
    // point o and bounding box b; the two cones overlap when there exist two vectors,
    // one in each cone, that are orthogonal to each other.
    public bool overlap(float3 o, BoundingBox b, float distToBox,
                        out float minAngleRange, out float maxAngleRange)
    {
        // initialize angle bounds
        minAngleRange = 0.0f;
        maxAngleRange = M_PI_2;

        // there's overlap if this cone's halfAngle is greater than 90 degrees, or
        // if the box contains the view cone origin (since the view cone is invalid)
        if (halfAngle >= M_PI_2 || distToBox < FLT_EPSILON)
        {
            return true;
        }

        // compute the view cone axis
        float3 c = b.getCentroid();
        float3 viewConeAxis = c - o;
        float l = length(viewConeAxis);
        viewConeAxis /= l;

        // check for overlap between the view cone axis and this cone
        float dAxisAngle = acos(max(-1.0, min(1.0, dot(axis, viewConeAxis)))); // [0, 180]
        if (inRange(M_PI_2, dAxisAngle - halfAngle, dAxisAngle + halfAngle))
        {
            return true;
        }

        // check if the view cone origin lies outside this cone's bounding sphere;
        // if it does, compute the view cone halfAngle and check for overlap
        if (l > radius)
        {
            float viewConeHalfAngle = asin(radius / l);
            float halfAngleSum = halfAngle + viewConeHalfAngle;
            minAngleRange = dAxisAngle - halfAngleSum;
            maxAngleRange = dAxisAngle + halfAngleSum;
            return halfAngleSum >= M_PI_2 ? true : inRange(M_PI_2, minAngleRange, maxAngleRange);
        }

        // the view cone origin lies inside the box's bounding sphere, so check if
        // the plane defined by the view cone axis intersects the box; if it does, then
        // there's overlap since the view cone has a halfAngle greater than 90 degrees
        float3 e = b.pMax - c;
        float d = dot(e, abs(viewConeAxis)); // max projection length onto axis
        float s = l - d;
        if (s <= 0.0)
        {
            return true;
        }

        // compute the view cone halfAngle by projecting the max extents of the box
        // onto the plane, and check for overlap
        d = projectToPlane(viewConeAxis, e);
        float viewConeHalfAngle = atan2(d, s);
        float halfAngleSum = halfAngle + viewConeHalfAngle;
        minAngleRange = dAxisAngle - halfAngleSum;
        maxAngleRange = dAxisAngle + halfAngleSum;
        return halfAngleSum >= M_PI_2 ? true : inRange(M_PI_2, minAngleRange, maxAngleRange);
    }
};

public float3 rotate(float3 u, float3 v, float theta)
{
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    float3 w = length(cross(u, v));
    float3 oneMinusCosThetaW = (1.0 - cosTheta) * w;

    float3x3 R;
    R[0][0] = cosTheta + oneMinusCosThetaW[0] * w[0];
    R[0][1] = oneMinusCosThetaW[1] * w[0] - sinTheta * w[2];
    R[0][2] = oneMinusCosThetaW[2] * w[0] + sinTheta * w[1];
    R[1][0] = oneMinusCosThetaW[0] * w[1] + sinTheta * w[2];
    R[1][1] = cosTheta + oneMinusCosThetaW[1] * w[1];
    R[1][2] = oneMinusCosThetaW[2] * w[1] - sinTheta * w[0];
    R[2][0] = oneMinusCosThetaW[0] * w[2] - sinTheta * w[1];
    R[2][1] = oneMinusCosThetaW[1] * w[2] + sinTheta * w[0];
    R[2][2] = cosTheta + oneMinusCosThetaW[2] * w[2];

    return mul(R, u);
}

public BoundingCone mergeBoundingCones(BoundingCone coneA, BoundingCone coneB,
                                       float3 originA, float3 originB,
                                       float3 newOrigin)
{
    BoundingCone cone;
    if (coneA.isValid() && coneB.isValid())
    {
        float3 axisA = coneA.axis;
        float3 axisB = coneB.axis;
        float halfAngleA = coneA.halfAngle;
        float halfAngleB = coneB.halfAngle;
        float3 dOriginA = newOrigin - originA;
        float3 dOriginB = newOrigin - originB;
        cone.radius = sqrt(max(coneA.radius * coneA.radius + dot(dOriginA, dOriginA),
                               coneB.radius * coneB.radius + dot(dOriginB, dOriginB)));

        if (halfAngleB > halfAngleA)
        {
            float3 tmpAxis = axisA;
            axisA = axisB;
            axisB = tmpAxis;

            float tmpHalfAngle = halfAngleA;
            halfAngleA = halfAngleB;
            halfAngleB = tmpHalfAngle;
        }

        float theta = acos(max(-1.0, min(1.0, dot(axisA, axisB))));
        if (min(theta + halfAngleB, M_PI) <= halfAngleA)
        {
            // right cone is completely inside left cone
            cone.axis = axisA;
            cone.halfAngle = halfAngleA;
            return cone;
        }

        // merge cones by first computing the spread angle of the cone to cover both cones
        float oTheta = (halfAngleA + theta + halfAngleB) / 2.0;
        if (oTheta >= M_PI)
        {
            cone.axis = axisA;
            return cone;
        }

        float rTheta = oTheta - halfAngleA;
        cone.axis = rotate(axisA, axisB, rTheta);
        cone.halfAngle = oTheta;
    }
    else if (coneA.isValid())
    {
        cone = coneA;
    }
    else if (coneB.isValid())
    {
        cone = coneB;
    }
    else
    {
        cone.halfAngle = -M_PI;
    }

    return cone;
}
