//TEST:SIMPLE(filecheck=CHECK): -target spirv
// CHECK: OpEntryPoint

module test;

public enum class MaterialID : uint { invalid = 0xffffffff };

public struct Material : IDifferentiable
{
    float x;
}

public struct Hit
{
    MaterialID material;
}

public struct Scene
{
    StructuredBuffer<Material> materials;
    RWStructuredBuffer<Material> grads;

    [Differentiable]
    Material load(MaterialID id) { return materials[uint(id)]; }

    void accumulate(MaterialID id, Material d) { grads[uint(id)].x += d.x; }

    [Differentiable, BackwardDerivative(_get_material_bwd)]
    public Material get_material(MaterialID id) { return load(id); }

    public void _get_material_bwd(MaterialID id, Material d) { accumulate(id, d); }

    [Differentiable]
    public Material get_material(Hit hit) { return get_material(hit.material); }
}

[Differentiable]
float trace(const Scene scene, Hit hit)
{
    Material m = scene.get_material(hit);
    return m.x;
}


[shader("compute")]
void main(
    uniform Scene scene,
    uniform StructuredBuffer<uint> input,
    uniform RWStructuredBuffer<float> output,
    uniform RWStructuredBuffer<float> grads
)
{
    Hit hit;
    hit.material = MaterialID(input[0]);
    output[0] = trace(scene, hit);
    bwd_diff(trace)(scene, hit, grads[0]);
}