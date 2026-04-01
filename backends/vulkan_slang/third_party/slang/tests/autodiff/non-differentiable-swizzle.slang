//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none

// This test just needs to compile without crashing.

// CUDA: __device__ void s_bwd_myKernel_0

[Differentiable]
uint3 load_foo(int32_t g_idx, TensorView<int32_t> indices)
{
    return {
        indices[uint2(g_idx, 0)],
        indices[uint2(g_idx, 1)],
        indices[uint2(g_idx, 2)]
    };
}

[Differentiable]
Triangle load_triangle(DiffTensorView vertices, uint3 index_set)
{
    float3[3] triangle = {
        read_t3_float3(index_set.x, vertices),
        read_t3_float3(index_set.y, vertices),
        read_t3_float3(index_set.z, vertices)
    };
    return { triangle };
}

struct Triangle : IDifferentiable
{
    float3[3] verts;
}

[Differentiable]
float3 read_t3_float3(uint32_t idx, DiffTensorView t3)
{
    return float3(t3[uint2(idx, 0)],
                  t3[uint2(idx, 1)],
                  t3[uint2(idx, 2)]);
}

[Differentiable] TriangleFoo load_triangle_foo(int32_t g_idx, DiffTensorView vertices, TensorView<int32_t> indices, DiffTensorView vertex_color) 
{ 
    uint3 index_set = load_foo(g_idx, indices); 
    Triangle triangle = load_triangle(vertices, index_set); 
    float3 c0 = read_t3_float3(index_set.x, vertex_color); 
    float3 c1 = read_t3_float3(index_set.y, vertex_color); 
    float3 c2 = read_t3_float3(index_set.z, vertex_color);
    TriangleFoo result = { triangle, 0.f, {c0, c1, c2} }; 
    return result; 
} 

struct TriangleFoo : IDifferentiable 
{ 
    Triangle triangle; 
    float density; 
    float3 vertex_color[3];
};

[AutoPyBindCUDA]
[Differentiable]
[CudaKernel]
void myKernel(DiffTensorView vertices, TensorView<int32_t> indices, DiffTensorView vertex_color)
{
    if (cudaThreadIdx().x > 0)
        return;

    TriangleFoo.Differential dp_g = TriangleFoo.dzero();

    bwd_diff(load_triangle_foo)(cudaThreadIdx().x, vertices, indices, vertex_color, dp_g);
}
