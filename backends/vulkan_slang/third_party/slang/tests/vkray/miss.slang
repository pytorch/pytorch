//TEST:CROSS_COMPILE: -profile glsl_460+GL_EXT_ray_tracing -stage miss -entry main -target spirv-assembly

struct ShadowRay
{
    float hitDistance;
};

void main(in out ShadowRay ray)
{
	ray.hitDistance = 10000.0f;
}
