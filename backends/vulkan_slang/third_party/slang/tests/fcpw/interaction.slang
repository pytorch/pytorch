implementing fcpw;
__include math_constants;

public struct Interaction
{
    public float3 p;   // interaction point associated with query
    public float3 n;   // normal at interaction point
    public float2 uv;  // uv coordinates of interaction point
    public float d;    // distance to interaction point
    public uint index; // index of primitive/silhouette associated with interaction point

    // constructor
    public __init()
    {
        p = float3(0.0, 0.0, 0.0);
        n = float3(0.0, 0.0, 0.0);
        uv = float2(0.0, 0.0);
        d = FLT_MAX;
        index = UINT_MAX;
    }
};
