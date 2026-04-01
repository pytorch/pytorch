//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//DISABLE_TEST:SIMPLE(filecheck=CHK):-target hlsl -stage compute -entry computeMain -report-checkpoint-intermediates

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer

RWStructuredBuffer<float> outputBuffer;

struct PathState
{
    uint depth;
    bool terminated;

    bool isHit() { return !terminated; }
    bool isTerminated() { return terminated; }
};

struct PathResult : IDifferentiable
{
    float thp;
    float L;
}
struct VisibilityQuery
{
    bool test();
}

struct ClosestHitQuery
{
    bool test();
}
void generatePath(uint pathID, out PathState path)
{
    path.terminated = false;
    path.depth = 0;
}

[BackwardDifferentiable]
float lightEval(uint depth)
{
    if (depth == 1)
    {
        return 5.0f;
    }
    else
    {
        return 0.0f;
    }
}

struct MaterialParam : IDifferentiable
{
    float roughness;
}

[BackwardDifferentiable]
MaterialParam getParam(uint id)
{
    MaterialParam p;
    p.roughness = 0.5f;
    return p;
}

[ForwardDerivativeOf(getParam)]
DifferentialPair<MaterialParam> d_getParam(uint id)
{
    MaterialParam p;
    p.roughness = 0.5f;
    MaterialParam.Differential d;
    d.roughness = 1.0f;
    return diffPair(p, d);
}

[BackwardDerivativeOf(getParam)]
void d_getParam(uint id, MaterialParam.Differential diff)
{
    outputBuffer[id] += diff.roughness;
}

//CHK-DAG: note: checkpointing context of 8 bytes associated with function: 'updatePathThroughput'
//CHK-DAG: note: 8 bytes (PathResult_0) used to checkpoint the following item:
[BackwardDifferentiable]
void updatePathThroughput(inout PathResult path, const float weight)
{
    path.thp *= weight;
}

struct BSDFSample : IDifferentiable
{
    float val;
}

[BackwardDifferentiable]
bool bsdfGGXSample(const MaterialParam bsdfParams, out BSDFSample result)
{
    result.val = bsdfParams.roughness;
    return true;
}

[BackwardDifferentiable]
bool generateScatterRay(const MaterialParam bsdfParams, inout PathState path, inout PathResult pathRes)
{
    BSDFSample result;
    bool valid = bsdfGGXSample(bsdfParams, result);
    return generateScatterRay(result, bsdfParams, path, pathRes, valid);
}

/** Generates a new scatter ray using BSDF importance sampling.
    \param[in] sd Shading data.
    \param[in] mi Material instance at the shading point.
    \param[in,out] path The path state.
    \return True if a ray was generated, false otherwise.
*/
[BackwardDifferentiable]
bool generateScatterRay(const BSDFSample bs, const MaterialParam bsdfParams, inout PathState path, inout PathResult pathRes, bool valid)
{
    if (valid) valid = generateScatterRay(bs, bsdfParams, path, pathRes);
    return valid;
}

/** Generates a new scatter ray given a valid BSDF sample.
    \param[in] bs BSDF sample (assumed to be valid).
    \param[in] sd Shading data.
    \param[in] mi Material instance at the shading point.
    \param[in,out] path The path state.
    \return True if a ray was generated, false otherwise.
*/

//CHK-DAG: note: checkpointing context of 16 bytes associated with function: 'generateScatterRay'
[BackwardDifferentiable]
bool generateScatterRay(const BSDFSample bs, const MaterialParam bsdfParams, inout PathState path, inout PathResult pathRes)
{
    //CHK-DAG: note: 8 bytes (s_bwd_prop_updatePathThroughput_Intermediates_0) used to checkpoint the following item:
    //CHK-DAG: note: 8 bytes (PathResult_0) used to checkpoint the following item:
    updatePathThroughput(pathRes, bs.val);
    return true;
}

[BackwardDifferentiable]
void handleHit(inout PathState path, inout PathResult rs, inout VisibilityQuery vq)
{
    var param = getParam(0);

    bool lastVertex = param.roughness > 0.8;
    if (lastVertex)
    {
        path.terminated = true;
        return;
    }
     
    generateScatterRay(param, path, rs);

    rs.L = rs.thp * lightEval(path.depth);

    // Decide on next hit
    if (path.depth < 1)
        path.terminated = false;
    else
        path.terminated = true;
}

[BackwardDifferentiable]
float bsdfEval(const MaterialParam mparam)
{
    return mparam.roughness;
}

[BackwardDifferentiable]
void nextHit(inout PathState path, inout PathResult rs, inout ClosestHitQuery cq)
{
    path.depth = path.depth + 1;
}

[BackwardDifferentiable]
void handleMiss(inout PathState path, inout PathResult rs)
{
    rs.L = 0.0f;
    path.terminated = true;
}

[BackwardDifferentiable]
bool tracePath(uint pathID, out PathState path, inout PathResult pathRes)
{
    generatePath(pathID, path);

    float thp = pathRes.thp;
    float L = pathRes.L;

    for (int i = 0; i < 3; ++i)
    {
        if (path.isHit())
        {
            VisibilityQuery vq;
            handleHit(path, pathRes, vq);

            if (path.isTerminated()) break;

            ClosestHitQuery chq;
            nextHit(path, pathRes, chq);
        }
        else
        {
            handleMiss(path, pathRes);
        }
    }
    
    return true;
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    {
        PathResult pathRes;
        pathRes.L = 1.f;
        pathRes.thp = 1.f;

        PathResult.Differential pathResD;
        pathResD.L = 1.0f;
        pathResD.thp = 0.f;

        var dpx = diffPair(pathRes, pathResD);
        __bwd_diff(tracePath)(1, dpx); // Expect: 5.0 in outputBuffer[3]
    }
}

//CHK-NOT: note