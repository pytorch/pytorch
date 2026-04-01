//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type -profile cs_5_1 -dx12

//TEST_INPUT:ubuffer(data=[0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0], stride=4):out,name=endpointDifferentialBuffer
RWStructuredBuffer<float> endpointDifferentialBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=endpointDifferentialBufferInt
RWStructuredBuffer<int> endpointDifferentialBufferInt;

//TEST_INPUT:ubuffer(data=[0.3 0.7 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0], stride=4):name=endpointBuffer
RWStructuredBuffer<float> endpointBuffer;
//TEST_INPUT:ubuffer(data=[1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0], stride=4):name=colorBuffer
RWStructuredBuffer<float> colorBuffer;

typedef float Color;

struct PRNG
{
    __init(uint seed)
    {
        this.state = seed;
    }

    [mutating] uint next()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }

    [mutating] float nextFloat1D()
    {
        return float(next()) / float(4294967295.0);
    }

    uint state;
};

struct LineSegment : IDifferentiable
{
    float x0;
    float x1;

    Color color;

    [BackwardDifferentiable]
    __init(float _x0, float _x1, Color _color)
    {
        x0 = _x0;
        x1 = _x1;
        color = _color;
    }
};

struct Intersection : IDifferentiable
{
    LineSegment ls;
    float x;
    bool isIntersected;
    float wt;

    [BackwardDifferentiable]
    __init(LineSegment _ls, float _x, bool _isIntersected, float _wt)
    {
        this.ls = _ls;
        this.x = _x;
        this.isIntersected = _isIntersected;
        this.wt = _wt;
    }
};

[BackwardDerivative(d_loadLineSegment)]
[ForwardDerivative(fwd_loadLineSegment)]
LineSegment loadLineSegment(uint id)
{
    return {endpointBuffer[id * 2], endpointBuffer[id * 2 + 1], colorBuffer[id]};
}

[BackwardDerivative(d_fwd_loadLineSegment)]
DifferentialPair<LineSegment> fwd_loadLineSegment(uint id)
{
    return DifferentialPair<LineSegment>(loadLineSegment(id), LineSegment.dzero());
}

void accumulateDifferentialFixedPoint(
    RWStructuredBuffer<int> buffer, 
    uint index,
    float.Differential df,
    float scale = 1000000.f)
{
    InterlockedAdd(buffer[index], (int)round(df * scale));
}

void d_loadLineSegment(uint id, LineSegment.Differential d_ls)
{
    accumulateDifferentialFixedPoint(endpointDifferentialBufferInt, id * 2, d_ls.x0);
    accumulateDifferentialFixedPoint(endpointDifferentialBufferInt, id * 2 + 1, d_ls.x1);
}

void d_fwd_loadLineSegment(uint id, DifferentialPair<LineSegment>.Differential dp_ls)
{
    accumulateDifferentialFixedPoint(endpointDifferentialBufferInt, id * 2, dp_ls.p.x0);
    accumulateDifferentialFixedPoint(endpointDifferentialBufferInt, id * 2 + 1, dp_ls.p.x1);
}

int getIntersectionID(float x)
{
    // Line segments are ordered by z-index so return the first intersection.
    for (int id = 0; id < 2; id++)
    {
        LineSegment ls = loadLineSegment(id);
        if (x > ls.x0 && x < ls.x1)
            return id;
    }
    return -1;
}

[BackwardDifferentiable]
Intersection intersect(float x)
{
    int id = getIntersectionID(x);
    if (id >= 0)
        return Intersection(loadLineSegment((uint)id), x, true, 1.0);

    return Intersection(LineSegment(0, 0, 0), x, false, 0.0);
}

[BackwardDifferentiable]
float shadeIntersection(Intersection isect)
{
    return isect.ls.color;
}

float sample1DNormal(inout PRNG prng, float mu, float sigma)
{
    float u = prng.nextFloat1D();
    float v = prng.nextFloat1D();
    return mu + (sqrt(-2 * log(u))*cos(2*3.1415*v) * sigma);
}

[BackwardDifferentiable]
float pdf1DNormal(no_diff float x, float mu, no_diff float sigma)
{  
    float k = ((x - mu) / sigma);
    return exp(-0.5 * (k * k)) / (sigma * 2.506628);
}

float boundaryTerm(Intersection isect)
{
    if (!isect.isIntersected)
        return 100.0; // Large default value for missed rays.
    
    float leftDist = abs(isect.x - isect.ls.x0);
    float rightDist = abs(isect.ls.x1 - isect.x);

    if (leftDist > rightDist)
        return rightDist * 30.f;
    else
        return leftDist * 30.f;
}

[BackwardDifferentiable]
DifferentialPair<float> infinitesimal(DifferentialPair<float> x)
{
    return diffPair(x.p - detach(x.p), x.d - detach(x.d));
}

[BackwardDifferentiable]
float harmonicWeight(Intersection isect, no_diff Intersection aux_isect)
{
    float x_dist = isect.x - aux_isect.x;
    float k = 1.0 / (((x_dist * x_dist) + no_diff(boundaryTerm(aux_isect))));
    return k;
}

[BackwardDifferentiable]
float attachToGeometry(Intersection isect)
{
    float leftWt = detach(isect.ls.x1 - isect.x);
    float rightWt = detach(isect.x - isect.ls.x0);

    return (leftWt * isect.ls.x0 + rightWt * isect.ls.x1) / (leftWt + rightWt);
}

[BackwardDifferentiable]
float warp(Intersection isect, inout PRNG prng)
{
    float totalWeight = 0.f;
    float totalWarpedPoint = 0.f;

    float aux_sigma = 0.01;

    for (int i = 0; i < 32; i++)
    {
        float y = no_diff(sample1DNormal(prng, isect.x, aux_sigma));
        float y_flipped = 2 * isect.x - y;

        Intersection aux_isect_left = intersect(y);

        if (aux_isect_left.isIntersected)
        {
            float pdf = pdf1DNormal(y, isect.x, aux_sigma);
            float wt = harmonicWeight(isect, aux_isect_left) * (pdf / detach(pdf));
            totalWarpedPoint += attachToGeometry(aux_isect_left) * wt;
            totalWeight += wt;
        }
        
        Intersection aux_isect_right = intersect(detach(y_flipped));

        if (aux_isect_right.isIntersected)
        {
            float pdf = pdf1DNormal(y_flipped, isect.x, aux_sigma);
            float wt = harmonicWeight(isect, aux_isect_right) * (pdf / detach(pdf));
            totalWarpedPoint += attachToGeometry(aux_isect_right) * wt;
            totalWeight += wt;
        }
    }

    return totalWarpedPoint / totalWeight;
}

[BackwardDifferentiable]
Intersection warpedIntersect(float x, inout PRNG prng)
{   
    // TODO: For now the jacobian here is 1.0, 
    // but we will need to adjust the warp by the jacobian for
    // more complex intersection models.
    //
    Intersection isect = intersect(x);

    Intersection.Differential d_isect = Intersection.Differential.dzero();
    d_isect.x = 1.0;

    var dpwarp = infinitesimal(
        __fwd_diff(warp)(diffPair(isect, d_isect), prng));

    isect.x = detach(isect.x) + dpwarp.p;
    isect.wt = isect.wt * (1 + dpwarp.d);

    return isect;
}

[BackwardDifferentiable]
float renderSample(inout PRNG prng) 
{
    float u = no_diff(prng.nextFloat1D());
    
    float leftBound = 0.0;
    float rightBound = 1.0;

    float sample = leftBound * u + rightBound * (1 - u);
    float weight = 1.0/(rightBound - leftBound);
 
    Intersection isect = warpedIntersect(sample, prng);

    return shadeIntersection(isect) * isect.wt;
}

[numthreads(1000, 1, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID,)
{
    uint seed = (threadIdx.x * threadIdx.x) * 30 + 3;
    PRNG prng = PRNG(seed);
    
    float d_color = 1.0 / 1000.0;
    __bwd_diff(renderSample)(prng, d_color);

    AllMemoryBarrierWithGroupSync();
    
    // Convert to floating point (but with 2 fewer digits of precision to 
    // avoid platform-specific differences in floating point precision)
    // 
    if (threadIdx.x < 10)
        endpointDifferentialBuffer[threadIdx.x] =
            ((endpointDifferentialBufferInt[threadIdx.x]/1000) / 1000000.f) * 1000.f;

// Note that this specific derivative estimation method is biased, so the
// expected results are approximate. (We've fixed the RNG seed to generate
// repeatable results)
//
// Expect: Approximately -1.0 in endpointDifferentialBuffer[0]
// Expect: Approximately 1.0 in endpointDifferentialBuffer[1]
//
// Expect: Approximately 0.0 in endpointDifferentialBuffer[2]
// Expect: Approximately 0.0 in endpointDifferentialBuffer[3]
//
}