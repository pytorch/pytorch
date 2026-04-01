implementing fcpw;
__include ray;
__include math_constants;
__include interaction;
__include bounding_volumes;

public float3x3 extractLinearTransform(float3x4 t)
{
    return float3x3(t[0].xyz, t[1].xyz, t[2].xyz);
}

public float3 extractTranslation(float3x4 t)
{
    return float3(t[0][3], t[1][3], t[2][3]);
}

public float3 transformPoint(float3x4 t, float3 p)
{
    return mul(t, float4(p, 1.0)).xyz;
}

public Ray transformRay(float3x4 t, Ray r)
{
    float3 o = transformPoint(t, r.o);
    float3 d = transformPoint(t, r.o + r.d * (r.tMax < FLT_MAX ? r.tMax : 1.0)) - o;
    float dNorm = length(d);

    return Ray(o, d / dNorm, r.tMax < FLT_MAX ? dNorm : FLT_MAX);
}

public BoundingSphere transformSphere(float3x4 t, BoundingSphere s)
{
    float3 c = transformPoint(t, s.c);
    float r2 = FLT_MAX;
    if (s.r2 < FLT_MAX)
    {
        float3 d = transformPoint(t, s.c + float3(sqrt(s.r2), 0.0, 0.0)) - c;
        r2 = dot(d, d);
    }

    return BoundingSphere(c, r2);
}

public void transformInteraction(float3x4 t, float3x4 tInv, float3 x,
                                 bool overwriteDistance, inout Interaction i)
{
    float3 p = transformPoint(t, i.p);
    float3 n = normalize(mul(transpose(extractLinearTransform(tInv)), i.n));

    i.p = p;
    i.n = n;
    if (overwriteDistance)
    {
        i.d = length(p - x);
    }
}
