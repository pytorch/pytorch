//TEST_IGNORE_FILE:
implementing "bsdf-sample";

struct ShadingData
{
    float3 V;
    float3 N;
    float3 T;
    float3 B;

    float3 fromLocal(float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
    }

    float3 toLocal(float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
    }
};

struct Auto_Bwd_ScatterSample : IDifferentiable
{
    float3 wo;
    float pdf;
    float3 weight;
};

struct Auto_Bwd_BSDFParameters : IDifferentiable
{
    float3 albedo;
    float roughness;
};

[BackwardDifferentiable]
bool bsdfGGXSample(in ShadingData sd, in Auto_Bwd_BSDFParameters params, out Auto_Bwd_ScatterSample result)
{
    float3 wiLocal = no_diff(sd.toLocal(sd.V));
    float2 u = float2(0.8, 0.3);

    if (wiLocal.z < 1e-6)
    {
        unused(result);
        return false;
    }

    // Taken from Rendering.Materials.Microfacet. Follows the Walter et al. EGSR07 BTDF paper
    float alphaSqr = params.roughness * params.roughness;
    float phi = u.y * (2 * 3.1415926);
    float tanThetaSqr = alphaSqr * u.x / (1 - u.x);
    float cosTheta = 1 / sqrt(1 + tanThetaSqr);
    float r = sqrt(max(1 - cosTheta * cosTheta, 0));

    float3 hLocal = float3(cos(phi) * r, sin(phi) * r, cosTheta); // half-vector local space
    float wiDotH = dot(wiLocal, hLocal);
    float3 woLocal = 2 * hLocal * wiDotH - wiLocal; // outgoing vector local space

    float pdf = bsdfGGXPDF(hLocal, params) / (4.f * wiDotH);
    result.wo = no_diff(sd.fromLocal(woLocal)); // wo to world.
    result.pdf = detach(pdf);
    result.weight = evalGGXDivByPDF(wiLocal, woLocal, hLocal, params) * pdf / detach(pdf);

    return woLocal.z > 1e-6;
}

[BackwardDifferentiable]
float3 F(float3 f0, float3 f90, float cosTheta)
{
    return f0 + (f90 - f0) * pow(max(1 - cosTheta, 0.f), 5.f);
}

[BackwardDifferentiable]
float evalLambdaGGX(float alphaSqr, float cosTheta)
{
    float cosThetaSqr = cosTheta * cosTheta;
    float tanThetaSqr = max(1 - cosThetaSqr, 0) / cosThetaSqr;
    return 0.5 * (-1 + sqrt(1 + alphaSqr * tanThetaSqr));
}

[BackwardDifferentiable]
float G(float alpha, float cosThetaI, float cosThetaO)
{
    float alphaSqr = alpha * alpha;
    float lambdaI = evalLambdaGGX(alphaSqr, cosThetaI);
    float lambdaO = evalLambdaGGX(alphaSqr, cosThetaO);
    return 1.0 / (1 + lambdaI + lambdaO);
}

[BackwardDifferentiable]
float3 evalGGXDivByPDF(in float3 wi, in float3 wo, in float3 h, in Auto_Bwd_BSDFParameters params)
{
    const float3 F0Color = params.albedo;
    let F90Color = float3(1.0, 1.0, 1.0);
    return F(F0Color, F90Color, dot(wi, h)) * G(params.roughness, wi.z, wo.z) * dot(wi, h) / (wi.z * h.z);
}

[BackwardDifferentiable]
float bsdfGGXPDF(in float3 hLocal, in Auto_Bwd_BSDFParameters params)
{
    float cosTheta = hLocal.z;

    float alpha = params.roughness;
    float a2 = alpha * alpha;
    float d = ((cosTheta * a2 - cosTheta) * cosTheta + 1);

    return (a2 / (d * d * 3.1415926)) * cosTheta;
}
