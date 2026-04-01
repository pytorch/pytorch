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

struct RayPayload
{
    float4 color;
};

uniform RWTexture2D resultTexture;
uniform RaytracingAccelerationStructure sceneBVH;
uniform StructuredBuffer<Primitive> primitiveBuffer;
uniform Uniforms uniforms;

[shader("raygeneration")]
void rayGenShader()
{
    uint2 threadIdx = DispatchRaysIndex().xy;
    if (threadIdx.x >= (int)uniforms.screenWidth) return;
    if (threadIdx.y >= (int)uniforms.screenHeight) return;

    float frameWidth = uniforms.screenWidth / uniforms.screenHeight * uniforms.frameHeight;
    float imageY = (threadIdx.y / uniforms.screenHeight - 0.5f) * uniforms.frameHeight;
    float imageX = (threadIdx.x / uniforms.screenWidth - 0.5f) * frameWidth;
    float imageZ = uniforms.focalLength;
    float3 rayDir = normalize(uniforms.cameraDir.xyz*imageZ - uniforms.cameraUp.xyz * imageY + uniforms.cameraRight.xyz * imageX);

    // Trace the ray.
    RayDesc ray;
    ray.Origin = uniforms.cameraPosition.xyz;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    RayPayload payload = { float4(0, 0, 0, 0) };
    TraceRay(sceneBVH, RAY_FLAG_NONE, ~0, 0, 0, 0, ray, payload);

    resultTexture[threadIdx.xy] = payload.color;
}

[shader("miss")]
void missShader(inout RayPayload payload)
{
    payload.color = float4(0, 0, 0, 1);
}

[shader("closesthit")]
void closestHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    float3 hitLocation = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    float3 shadowRayDir = uniforms.lightDir.xyz;

    RayDesc ray;
    ray.Origin = hitLocation;
    ray.Direction = shadowRayDir;
    ray.TMin = 0.001;
    ray.TMax = 10000.0;
    RayPayload shadowPayload = { float4(0, 0, 0, 0) };
    TraceRay(sceneBVH, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 1, 0, 0, ray, shadowPayload);
    float shadow = 1.0 - shadowPayload.color.x;

    let primitiveIndex = PrimitiveIndex();
    float3 normal = primitiveBuffer[primitiveIndex].getNormal();
    float3 color = primitiveBuffer[primitiveIndex].getColor();
    float ndotl = max(0.0, shadow * dot(normal, uniforms.lightDir.xyz));
    float intensity = ndotl * 0.7 + 0.3;
    payload.color = float4(color * intensity, 1.0f);
}

[shader("closesthit")]
void shadowRayHitShader(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.color = float4(1.0, 1.0, 1.0, 1.0);
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
