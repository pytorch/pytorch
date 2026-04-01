implementing fcpw;

public struct Ray
{
    public float3 o;    // ray origin
    public float3 d;    // ray direction
    public float3 dInv; // 1 over ray direction (coordinate-wise)
    public float tMax;  // max ray distance

    // constructor
    public __init(float3 o_, float3 d_, float tMax_)
    {
        o = o_;
        d = d_;
        dInv = float3(1.0, 1.0, 1.0) / d_;
        tMax = tMax_;
    }
};
