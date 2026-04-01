//TEST:SIMPLE(filecheck=CHK): -target spirv -entry computeMain

// Delegated constructors
struct Impl
{
    float x;

    __init(float val)
    {
        x = val;
    }

    __init()
    {
        float val = 2.0;

        // Shouldn't trigger a warning here
        return Impl(val);
    }
}

// Calling a method from a constructor to initialize fields
struct HitInfo
{
    float3 barycentrics;
    uint primitiveIndex;
    
    [[mutating]] void init(float2 hitBarycentrics, uint hitPrimitiveIndex)
    {
        barycentrics = { 1.0 - hitBarycentrics.x - hitBarycentrics.y, hitBarycentrics.x, hitBarycentrics.y };
        primitiveIndex = hitPrimitiveIndex;
    }

    __init(float2 hitBarycentrics, uint hitPrimitiveIndex)
    {
        init(hitBarycentrics, hitPrimitiveIndex);
    }

    __init(BuiltInTriangleIntersectionAttributes attr)
    {
        init(attr.barycentrics, PrimitiveIndex());
    }
}

//CHK-NOT: warning 41020
//CHK-NOT: warning 41021


[Shader("compute")]
[NumThreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
}

