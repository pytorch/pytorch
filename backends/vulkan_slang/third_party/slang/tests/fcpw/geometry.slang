implementing fcpw;
__include ray;
__include math_constants;
__include interaction;
__include bounding_volumes;

public interface IPrimitive
{
    // returns the bounding box of the primitive
    BoundingBox getBoundingBox();

    // returns the centroid of the primitive
    float3 getCentroid();

    // returns the normal of the primitive
    float3 getNormal();

    // returns the surface area of the primitive
    float getSurfaceArea();

    // intersects primitive with ray
    bool intersect(Ray r, bool checkForOcclusion, inout Interaction i);

    // intersects primitive with sphere
    bool intersect(BoundingSphere s, inout Interaction i);

    // finds closest point on primitive from sphere center
    bool findClosestPoint(BoundingSphere s, inout Interaction i);

    // samples point on primitive and returns sampling pdf
    float samplePoint(float2 randNums, out float2 uv, out float3 p, out float3 n);

    // returns the index of the primitive
    uint getIndex();
};

public bool intersectLineSegment(float3 pa, float3 pb,
                                 float3 ro, float3 rd, float rtMax, bool checkForOcclusion,
                                 inout float3 p, inout float3 n, inout float2 uv, inout float d)
{
    float3 u = pa - ro;
    float3 v = pb - pa;

    // return if line segment and ray are parallel
    float dv = cross(rd, v)[2];
    if (abs(dv) <= FLT_EPSILON)
    {
        return false;
    }

    // solve ro + t*rd = pa + s*(pb - pa) for t >= 0 && 0 <= s <= 1
    // s = (u x rd)/(rd x v)
    float ud = cross(u, rd)[2];
    float s = ud / dv;

    if (s >= 0.0 && s <= 1.0)
    {
        // t = (u x v)/(rd x v)
        float t = cross(u, v)[2] / dv;

        if (t >= 0.0 && t <= rtMax)
        {
            if (checkForOcclusion)
            {
                return true;
            }

            p = pa + s * v;
            n = normalize(float3(v[1], -v[0], 0.0));
            uv = float2(s, 0.0);
            d = t;
            return true;
        }
    }

    return false;
}

public float findClosestPointLineSegment(float3 pa, float3 pb, float3 x, out float3 p, out float t)
{
    float3 u = pb - pa;
    float3 v = x - pa;

    float c1 = dot(u, v);
    if (c1 <= 0.0)
    {
        t = 0.0;
        p = pa;
        return length(x - p);
    }

    float c2 = dot(u, u);
    if (c2 <= c1)
    {
        t = 1.0;
        p = pb;
        return length(x - p);
    }

    t = c1 / c2;
    p = pa + u * t;
    return length(x - p);
}

public struct LineSegment : IPrimitive
{
    public float3 pa;
    public float3 pb;
    public uint index;

    // returns the bounding box of the primitive
    public BoundingBox getBoundingBox()
    {
        float3 epsilon = float3(FLT_EPSILON, FLT_EPSILON, 0.0);
        return BoundingBox(min(pa, pb) - epsilon, max(pa, pb) + epsilon);
    }

    // returns the centroid of the primitive
    public float3 getCentroid()
    {
        return 0.5 * (pa + pb);
    }

    // returns the normal of the primitive
    public float3 getNormal()
    {
        float3 s = pb - pa;
        float3 n = float3(s.y, -s.x, 0.0);

        return normalize(n);
    }

    // returns the surface area of the primitive
    public float getSurfaceArea()
    {
        return length(pb - pa);
    }

    // intersects primitive with ray
    // NOTE: specialized to 2D (z coordinate == 0)
    public bool intersect(Ray r, bool checkForOcclusion, inout Interaction i)
    {
        bool didIntersect = intersectLineSegment(pa, pb, r.o, r.d, r.tMax, checkForOcclusion, i.p, i.n, i.uv, i.d);
        if (didIntersect)
        {
            i.index = index;
            return true;
        }

        return false;
    }

    // intersects primitive with sphere
    public bool intersect(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointLineSegment(pa, pb, s.c, i.p, i.uv[0]);
        if (d * d <= s.r2)
        {
            i.d = getSurfaceArea();
            i.index = index;
            return true;
        }

        return false;
    }

    // finds closest point on primitive from sphere center
    public bool findClosestPoint(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointLineSegment(pa, pb, s.c, i.p, i.uv[0]);
        if (d * d <= s.r2)
        {
            i.uv[1] = 0.0;
            i.d = d;
            i.index = index;
            return true;
        }

        return false;
    }

    // samples point on primitive and returns sampling pdf
    public float samplePoint(float2 randNums, out float2 uv, out float3 p, out float3 n)
    {
        float3 s = pb - pa;
        float area = length(s);
        float u = randNums[0];
        uv = float2(u, 0.0);
        p = pa + u * s;
        n = float3(s[1], -s[0], 0.0) / area;

        return 1.0 / area;
    }

    // returns the index of the primitive
    public uint getIndex()
    {
        return index;
    }
};

public bool intersectTriangle(float3 pa, float3 pb, float3 pc,
                              float3 ro, float3 rd, float rtMax, bool checkForOcclusion,
                              inout float3 p, inout float3 n, inout float2 uv, inout float d)
{
    // Möller–Trumbore intersection algorithm
    float3 v1 = pb - pa;
    float3 v2 = pc - pa;
    float3 q = cross(rd, v2);
    float det = dot(v1, q);

    // ray and triangle are parallel if det is close to 0
    if (abs(det) <= FLT_EPSILON)
    {
        return false;
    }
    float invDet = 1.0 / det;

    float3 r = ro - pa;
    float v = dot(r, q) * invDet;
    if (v < 0.0 || v > 1.0)
    {
        return false;
    }

    float3 s = cross(r, v1);
    float w = dot(rd, s) * invDet;
    if (w < 0.0 || v + w > 1.0)
    {
        return false;
    }

    float t = dot(v2, s) * invDet;
    if (t >= 0.0 && t <= rtMax)
    {
        if (checkForOcclusion)
        {
            return true;
        }

        p = pa + v1 * v + v2 * w;
        n = normalize(cross(v1, v2));
        uv = float2(1.0 - v - w, v);
        d = t;
        return true;
    }

    return false;
}

public float findClosestPointTriangle(float3 pa, float3 pb, float3 pc, float3 x, out float3 p, out float2 t)
{
    // source: real time collision detection
    // check if x in vertex region outside pa
    float3 ab = pb - pa;
    float3 ac = pc - pa;
    float3 ax = x - pa;
    float d1 = dot(ab, ax);
    float d2 = dot(ac, ax);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        // barycentric coordinates (1, 0, 0)
        t = float2(1.0, 0.0);
        p = pa;
        return length(x - p);
    }

    // check if x in vertex region outside pb
    float3 bx = x - pb;
    float d3 = dot(ab, bx);
    float d4 = dot(ac, bx);
    if (d3 >= 0.0 && d4 <= d3)
    {
        // barycentric coordinates (0, 1, 0)
        t = float2(0.0, 1.0);
        p = pb;
        return length(x - p);
    }

    // check if x in vertex region outside pc
    float3 cx = x - pc;
    float d5 = dot(ab, cx);
    float d6 = dot(ac, cx);
    if (d6 >= 0.0 && d5 <= d6)
    {
        // barycentric coordinates (0, 0, 1)
        t = float2(0.0, 0.0);
        p = pc;
        return length(x - p);
    }

    // check if x in edge region of ab, if so return projection of x onto ab
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        // barycentric coordinates (1 - v, v, 0)
        float v = d1 / (d1 - d3);
        t = float2(1.0 - v, v);
        p = pa + ab * v;
        return length(x - p);
    }

    // check if x in edge region of ac, if so return projection of x onto ac
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        // barycentric coordinates (1 - w, 0, w)
        float w = d2 / (d2 - d6);
        t = float2(1.0 - w, 0.0);
        p = pa + ac * w;
        return length(x - p);
    }

    // check if x in edge region of bc, if so return projection of x onto bc
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        // barycentric coordinates (0, 1 - w, w)
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        t = float2(0.0, 1.0 - w);
        p = pb + (pc - pb) * w;
        return length(x - p);
    }

    // x inside face region. Compute p through its barycentric coordinates (u, v, w)
    float denom = 1.0 / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    t = float2(1.0 - v - w, v);
    p = pa + ab * v + ac * w; //= u*a + v*b + w*c, u = va*denom = 1.0f - v - w
    return length(x - p);
}

public struct Triangle : IPrimitive
{
    public float3 pa;
    public float3 pb;
    public float3 pc;
    public uint index;

    // returns the bounding box of the primitive
    public BoundingBox getBoundingBox()
    {
        float3 epsilon = float3(FLT_EPSILON, FLT_EPSILON, FLT_EPSILON);
        return BoundingBox(min(min(pa, pb), pc) - epsilon, max(max(pa, pb), pc) + epsilon);
    }

    // returns the centroid of the primitive
    public float3 getCentroid()
    {
        return (pa + pb + pc) / 3.0;
    }

    // returns the surface area of the primitive
    public float getSurfaceArea()
    {
        return 0.5 * length(cross(pb - pa, pc - pa));
    }

    // returns the normal of the primitive
    public float3 getNormal()
    {
        float3 n = cross(pb - pa, pc - pa);

        return normalize(n);
    }

    // intersects primitive with ray
    public bool intersect(Ray r, bool checkForOcclusion, inout Interaction i)
    {
        bool didIntersect = intersectTriangle(pa, pb, pc, r.o, r.d, r.tMax, checkForOcclusion, i.p, i.n, i.uv, i.d);
        if (didIntersect)
        {
            i.index = index;
            return true;
        }

        return false;
    }

    // intersects primitive with sphere
    public bool intersect(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointTriangle(pa, pb, pc, s.c, i.p, i.uv);
        if (d * d <= s.r2)
        {
            i.d = getSurfaceArea();
            i.index = index;
            return true;
        }

        return false;
    }

    // finds closest point on primitive from sphere center
    public bool findClosestPoint(BoundingSphere s, inout Interaction i)
    {
        float d = findClosestPointTriangle(pa, pb, pc, s.c, i.p, i.uv);
        if (d * d <= s.r2)
        {
            i.d = d;
            i.index = index;
            return true;
        }

        return false;
    }

    // samples point on primitive and returns sampling pdf
    public float samplePoint(float2 randNums, out float2 uv, out float3 p, out float3 n)
    {
        n = cross(pb - pa, pc - pa);
        float area = length(n);
        float u1 = sqrt(randNums[0]);
        float u2 = randNums[1];
        float u = 1.0 - u1;
        float v = u2 * u1;
        float w = 1.0 - u - v;
        uv = float2(u, v);
        p = pa * u + pb * v + pc * w;
        n /= area;

        return 2.0 / area;
    }

    // returns the index of the primitive
    public uint getIndex()
    {
        return index;
    }
};

public interface ISilhouette
{
    // returns the centroid of the silhouette
    float3 getCentroid();

    // returns whether silhouette has two adjacent faces
    bool hasTwoAdjacentFaces();

    // returns normal of adjacent face
    float3 getNormal(uint fIndex);

    // finds closest silhouette point on primitive from sphere center
    bool findClosestSilhouettePoint(BoundingSphere s, bool flipNormalOrientation,
                                    float squaredMinRadius, float precision,
                                    inout Interaction i);

    // returns the index of the silhouette
    uint getIndex();
};

public struct NoSilhouette : ISilhouette
{
    public uint index;

    // returns the centroid of the silhouette
    public float3 getCentroid()
    {
        return float3(0.0, 0.0, 0.0);
    }

    // returns whether silhouette has two adjacent faces
    public bool hasTwoAdjacentFaces()
    {
        return false;
    }

    // returns normal of adjacent face
    public float3 getNormal(uint fIndex)
    {
        return float3(0.0, 0.0, 0.0);
    }

    // finds closest silhouette point on primitive from sphere center
    public bool findClosestSilhouettePoint(BoundingSphere s, bool flipNormalOrientation,
                                           float squaredMinRadius, float precision,
                                           inout Interaction i)
    {
        return false;
    }

    // returns the index of the silhouette
    public uint getIndex()
    {
        return UINT_MAX;
    }
};

public bool isSilhouetteVertex(float3 n0, float3 n1, float3 viewDir, float d, bool flipNormalOrientation, float precision)
{
    float sign = flipNormalOrientation ? 1.0 : -1.0;

    // vertex is a silhouette point if it is concave and the query point lies on the vertex
    if (d <= precision)
    {
        float det = n0.x * n1.y - n1.x * n0.y;
        return sign * det > precision;
    }

    // vertex is a silhouette point if the query point lies on the halfplane
    // defined by an adjacent line segment and the other segment is backfacing
    float3 viewDirUnit = viewDir / d;
    float dot0 = dot(viewDirUnit, n0);
    float dot1 = dot(viewDirUnit, n1);

    bool isZeroDot0 = abs(dot0) <= precision;
    if (isZeroDot0)
    {
        return sign * dot1 > precision;
    }

    bool isZeroDot1 = abs(dot1) <= precision;
    if (isZeroDot1)
    {
        return sign * dot0 > precision;
    }

    // vertex is a silhouette point if an adjacent line segment is frontfacing
    // w.r.t. the query point and the other segment is backfacing
    return dot0 * dot1 < 0.0;
}

public struct Vertex : ISilhouette
{
    public float3 p;
    public float3 n0;
    public float3 n1;
    public uint index;
    public uint hasOneAdjacentFace;

    // returns the centroid of the silhouette
    public float3 getCentroid()
    {
        return p;
    }

    // returns whether silhouette has two adjacent faces
    public bool hasTwoAdjacentFaces()
    {
        return hasOneAdjacentFace == 0;
    }

    // returns normal of adjacent face
    public float3 getNormal(uint fIndex)
    {
        if (fIndex == 0)
        {
            return n0;
        }

        return n1;
    }

    // finds closest silhouette point on primitive from sphere center
    public bool findClosestSilhouettePoint(BoundingSphere s, bool flipNormalOrientation,
                                           float squaredMinRadius, float precision,
                                           inout Interaction i)
    {
        if (squaredMinRadius >= s.r2)
        {
            return false;
        }

        // compute view direction
        float3 viewDir = s.c - p;
        float d = length(viewDir);
        if (d * d > s.r2)
        {
            return false;
        }

        // check if vertex is a silhouette point from view direction
        bool process = hasOneAdjacentFace == 1 ? true : false;
        if (!process)
        {
            process = isSilhouetteVertex(n0, n1, viewDir, d, flipNormalOrientation, precision);
        }

        if (process && d * d <= s.r2)
        {
            i.p = p;
            i.uv = float2(0.0, 0.0);
            i.d = d;
            i.index = index;
            return true;
        }

        return false;
    }

    // returns the index of the silhouette
    public uint getIndex()
    {
        return index;
    }
};

public bool isSilhouetteEdge(float3 pa, float3 pb, float3 n0, float3 n1, float3 viewDir,
                             float d, bool flipNormalOrientation, float precision)
{
    float sign = flipNormalOrientation ? 1.0 : -1.0;

    // edge is a silhouette if it is concave and the query point lies on the edge
    if (d <= precision)
    {
        float3 edgeDir = normalize(pb - pa);
        float signedDihedralAngle = atan2(dot(edgeDir, cross(n0, n1)), dot(n0, n1));
        return sign * signedDihedralAngle > precision;
    }

    // edge is a silhouette if the query point lies on the halfplane defined
    // by an adjacent triangle and the other triangle is backfacing
    float3 viewDirUnit = viewDir / d;
    float dot0 = dot(viewDirUnit, n0);
    float dot1 = dot(viewDirUnit, n1);

    bool isZeroDot0 = abs(dot0) <= precision;
    if (isZeroDot0)
    {
        return sign * dot1 > precision;
    }

    bool isZeroDot1 = abs(dot1) <= precision;
    if (isZeroDot1)
    {
        return sign * dot0 > precision;
    }

    // edge is a silhouette if an adjacent triangle is frontfacing w.r.t. the
    // query point and the other triangle is backfacing
    return dot0 * dot1 < 0.0;
}

public struct Edge : ISilhouette
{
    public float3 pa;
    public float3 pb;
    public float3 n0;
    public float3 n1;
    public uint index;
    public uint hasOneAdjacentFace;

    // returns the centroid of the silhouette
    public float3 getCentroid()
    {
        return 0.5 * (pa + pb);
    }

    // returns whether silhouette has two adjacent faces
    public bool hasTwoAdjacentFaces()
    {
        return hasOneAdjacentFace == 0;
    }

    // returns normal of adjacent face
    public float3 getNormal(uint fIndex)
    {
        if (fIndex == 0)
        {
            return n0;
        }

        return n1;
    }

    // finds closest silhouette point on primitive from sphere center
    public bool findClosestSilhouettePoint(BoundingSphere s, bool flipNormalOrientation,
                                           float squaredMinRadius, float precision,
                                           inout Interaction i)
    {
        if (squaredMinRadius >= s.r2)
        {
            return false;
        }

        // compute view direction
        float d = findClosestPointLineSegment(pa, pb, s.c, i.p, i.uv[0]);
        if (d * d > s.r2)
        {
            return false;
        }

        // check if edge is a silhouette from view direction
        bool process = hasOneAdjacentFace == 1 ? true : false;
        if (!process)
        {
            float3 viewDir = s.c - i.p;
            process = isSilhouetteEdge(pa, pb, n0, n1, viewDir, d, flipNormalOrientation, precision);
        }

        if (process && d * d <= s.r2)
        {
            i.uv[1] = 0.0;
            i.d = d;
            i.index = index;
            return true;
        }

        return false;
    }

    // returns the index of the silhouette
    public uint getIndex()
    {
        return index;
    }
};
