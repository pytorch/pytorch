//TEST:SIMPLE(filecheck=CHECK): -target spirv

// This is a test that checks that we can apply partial specialization to a function
// that takes existential parameters.

// CHECK: OpRayQueryProceedKHR
// CHECK: OpImageWrite

public interface IRandom {
    [mutating] uint32_t next_uint();
    [mutating] float next_float();
};

public struct TEA : IRandom {
    int val;
    public __init(uint32_t v0, uint32_t v1) {}
    [mutating] public uint32_t next_uint() {
        return val++;
    }
    [mutating] public float next_float() {
        return val++;
    }
};

public interface IScene {
    property RaytracingAccelerationStructure as;
};

// The Scene type contains a resource field, if dynamic dispatch code were generated
// for this type, we will get a compile error.
struct Scene : IScene {
    RaytracingAccelerationStructure as;
};

public interface IIntegrator {
    
    // This function takes two existential parameters, `scene` and `rng`.
    // if we call this function with `rng` being dynamic, and `scene` being static,
    // we should still be able to specialize the `sample` function with the statically known
    // type of `scene`.
    public float3 sample(IScene scene, RayDesc ray, IRandom rng);
};
namespace integrator {
    public struct NoShading : IIntegrator {
        public float3 sample(IScene scene, RayDesc _ray, IRandom rng) {
            return float3( 0.0f, 0.0f, 0.0f );
        }
    };

    struct Test {

        float4 sample(IScene scene, RayDesc _ray, IRandom rng, int pixel_aabb_uv, uint3 id) {
            float4 grad = { 0.0f, 0.0f, 0.0f, 1.0f };

            RayDesc ray = _ray; uint32_t depth = 0;
            RayQuery<0> rayQuery;
            while (rayQuery.Proceed()) {
                float rand = rng.next_float();

                IIntegrator integrator = integrator::NoShading();

                // Here `rng` is mutating in the loop, so its type is dynamic and we need
                // to generate dynamic dispatch code around it.
                // But this shouldn't result in `scene` being dynamic as well.
                // We should still be able to specialize `integrator.sample` with the statically
                // known type of `scene`.
                // If this doesn't happen, then the compiler will try to synthesize dynamic dispatch
                // logic for `scene` and fail to compile.
                let color_in = integrator.sample(scene, {}, rng);
                let color_out = integrator.sample(scene, {}, rng);
                grad.xyz += color_out;
            }
            return grad;
        }
    };
};

[[vk::binding(0, 0)]] RWTexture2D<float4> output;

[vk::constant_id(0)] const int WGS_X = 1;
[vk::constant_id(1)] const int WGS_Y = 1;
[shader("compute"), numthreads(WGS_X, WGS_Y, 1)]
void main(
    uint3 id : SV_DispatchThreadID
)
{
    IRandom rng = TEA(id.y * id.x, 1);
    Scene scene = { RaytracingAccelerationStructure(0) };

    integrator::Test integrator = integrator::Test();
    float4 grad = integrator.sample(scene, {}, rng, {}, id);
    output[id.xy] += float4(grad.xyz, grad.w);
}