// nested-parameter-block-3.slang
// 
// Same as nest-parameter-block-2, the only difference is that we added member methods
// to ParameterBlock element types to exercise the legalization logic of method calls.

//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -d3d12 -use-dxil -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -vk -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=CHECK): -metal -shaderobj -output-using-type -render-features argument-buffer-tier-2


struct CB
{
    uint4 value;
}

struct MaterialSystem
{
    CB cb;
    RWStructuredBuffer<uint4> data;
    uint4 readData() { return data[0]; }
}

struct Scene
{
    CB sceneCb;
    RWStructuredBuffer<uint4> data;
    ParameterBlock<MaterialSystem> material;
    uint4 readMaterialData() { return material.readData(); }
}

//TEST_INPUT: set scene = new Scene { { {1,2,3,4} }, ubuffer(data=[1 2 3 4], stride=4), new MaterialSystem {{ {1,2,3,4} }, ubuffer(data=[1 2 3 4], stride=4)} }
ParameterBlock<Scene> scene;

struct MyBuffer
{
    RWStructuredBuffer<uint> resultBuffer;
}
//TEST_INPUT: set pb2 = new MyBuffer { out ubuffer(data=[0 0 0 0], stride=4) }
ParameterBlock<MyBuffer> pb2;

// Main entry-point. Applies the transformation encoded by `transformer`
// to all elements in `buffer`.
[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 sv_dispatchThreadID : SV_DispatchThreadID)
{
    // CHECK: type: uint32_t
    // CHECK-NEXT: 4
    // CHECK-NEXT: 2
    pb2.resultBuffer[0] = scene.sceneCb.value.x + scene.data[0].x + scene.material.cb.value.x + scene.material.data[0].x;
    pb2.resultBuffer[1] = scene.readMaterialData().y;
}
