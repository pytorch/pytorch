// shaders.slang

struct Uniforms
{
    float screenWidth, screenHeight;
    float focalLength, frameHeight;
    float4 cameraDir;
    float4 cameraUp;
    float4 cameraRight;
    float4 cameraPosition;
    float4 lightDir;
};

struct Primitive
{
    float4 data0;
    float4 color;
    float3 getNormal() { return data0.xyz; }
    float3 getColor() { return color.xyz; }
};

bool traceRayFirstHit(
    RaytracingAccelerationStructure sceneBVH,
    float3 rayOrigin,
    float3 rayDir,
    out float t,
    out int primitiveIndex)
{
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.TMin = 0.01f;
    ray.Direction = rayDir;
    ray.TMax = 1e4f;
    RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    let rayFlags = RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES |
             RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

    q.TraceRayInline(
        sceneBVH,
        rayFlags,
        0xff,
        ray);
    q.Proceed();

    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        t = q.CommittedRayT();
        primitiveIndex = q.CommittedPrimitiveIndex();
        return true;
    }
    return false;
}

bool traceRayNearestHit(
    RaytracingAccelerationStructure sceneBVH,
    float3 rayOrigin,
    float3 rayDir,
    out float t,
    out int primitiveIndex)
{
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.TMin = 0.01f;
    ray.Direction = rayDir;
    ray.TMax = 1e4f;
    RayQuery<RAY_FLAG_NONE> q;
    let rayFlags = RAY_FLAG_NONE;

    q.TraceRayInline(
        sceneBVH,
        rayFlags,
        0xff,
        ray);

    q.Proceed();
    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        t = q.CommittedRayT();
        primitiveIndex = q.CommittedPrimitiveIndex();
        return true;
    }
    return false;
}

[shader("compute")]
[numthreads(16,16,1)]
void computeMain(
    uint3 threadIdx : SV_DispatchThreadID,
    uniform RWTexture2D resultTexture,
    uniform RaytracingAccelerationStructure sceneBVH,
    uniform StructuredBuffer<Primitive> primitiveBuffer,
    uniform Uniforms uniforms)
{
    if (threadIdx.x >= (int)uniforms.screenWidth) return;
    if (threadIdx.y >= (int)uniforms.screenHeight) return;

    float frameWidth = uniforms.screenWidth / uniforms.screenHeight * uniforms.frameHeight;
    float imageY = (threadIdx.y / uniforms.screenHeight - 0.5f) * uniforms.frameHeight;
    float imageX = (threadIdx.x / uniforms.screenWidth - 0.5f) * frameWidth;
    float imageZ = uniforms.focalLength;
    float3 rayDir = normalize(uniforms.cameraDir.xyz*imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageX);

    float4 resultColor = 0;

    int primitiveIndex;
    float intersectionT;
    if (traceRayNearestHit(sceneBVH, uniforms.cameraPosition.xyz, rayDir, intersectionT, primitiveIndex))
    {
        float3 hitLocation = uniforms.cameraPosition.xyz + rayDir * intersectionT;
        float3 shadowRayDir = uniforms.lightDir.xyz;
        float shadow = 1.0;
        float shadowIntersectionT;
        int shadowPrimitiveIndex;
        if (traceRayFirstHit(sceneBVH, hitLocation, shadowRayDir, shadowIntersectionT, shadowPrimitiveIndex))
        {
            shadow = 0.0f;
        }
        float3 normal = primitiveBuffer[primitiveIndex].getNormal();
        float3 color = primitiveBuffer[primitiveIndex].getColor();
        float ndotl = max(0.0, shadow * dot(normal, uniforms.lightDir.xyz));
        float intensity = ndotl * 0.7 + 0.3;
        resultColor = float4(color * intensity, 1.0f);
    }
    resultTexture[threadIdx.xy] = resultColor;
}

/// Vertex and fragment shader for displaying the final image.

[shader("vertex")]
float4 vertexMain(float2 position : POSITION)
    : SV_Position
{
    return float4(position, 0.5, 1.0);
}

[shader("fragment")]
float4 fragmentMain(
    float4 sv_position : SV_Position,
    uniform RWTexture2D t)
    : SV_Target
{
    return t.Load(sv_position.xy);
}
