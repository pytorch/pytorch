// trivial-copy-textures.slang

typedef uint4 Element;

// UNORDERED ACCESS VIEW
[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest3DUnordered(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform RWTexture3D<Element> resourceView,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;

    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);

    // Read from resourceView
    Element elem = resourceView[coord];

    // Write to resourceView (if possible)
    resourceView[coord] = Element(1);

    // Write something to testResults
    testResults[tid] = elem;
}

[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest2DUnordered(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform RWTexture2D<Element> resourceView,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;

    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);

    // Read from resourceView
    Element elem = resourceView[coord.xy];

    // Write to resourceView (if possible)
    resourceView[coord.xy] = Element(1);

    // Write something to testResults
    testResults[tid] = elem;
}

[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest1DUnordered(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform RWTexture1D<Element> resourceView,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;

    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);

    // Read from resourceView
    Element elem = resourceView[coord.x];

    // Write to resourceView (if possible)
    resourceView[coord.x] = Element(1);

    // Write something to testResults
    testResults[tid] = elem;
}

// SHADER RESOURCE VIEW
[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest3DShader(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform Texture3D<Element> resourceView,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;

    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);

    // Read from resourceView
    Element elem = resourceView[coord];

    // Write something to testResults
    testResults[tid] = elem;
}

[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest2DShader(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform Texture2D<Element> resourceView,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;

    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);

    // Read from resourceView
    Element elem = resourceView[coord.xy];

    // Write something to testResults
    testResults[tid] = elem;
}

[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest1DShader(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform Texture1D<Element> resourceView,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;

    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);

    // Read from resourceView
    Element elem = resourceView[coord.x];

    // Write something to testResults
    testResults[tid] = elem;
}

// RENDER TARGET AND DEPTH STENCIL VIEWS
// Per-vertex attributes to be assembled from bound vertex buffers.
struct AssembledVertex
{
    float3	position : POSITION;
    float3	color    : COLOR;
};

// Output of the vertex shader, and input to the fragment shader.
struct CoarseVertex
{
    float3 color;
};

// Output of the fragment shader
struct Fragment
{
    float4 color;
};

// Vertex  Shader

struct VertexStageOutput
{
    CoarseVertex    coarseVertex    : CoarseVertex;
    float4          sv_position     : SV_Position;
};

[shader("vertex")]
VertexStageOutput vertexMain(
    AssembledVertex assembledVertex)
{
    VertexStageOutput output;

    float3 position = assembledVertex.position;
    float3 color    = assembledVertex.color;

    output.coarseVertex.color = color;
    output.sv_position = float4(position, 1.0);

    return output;
}

// Fragment Shader

[shader("fragment")]
float4 fragmentMain() : SV_Target
{
    return float4(1.0, 2.0, 3.0, 4.0);
}


#if 0
typedef uint4 Element;

interface ITestCase
{
    Element doTest(uint3 coord);
}

// 1D + array, 2D + array + multisample + multisample array, cube + array

struct UAV_3D : ITestCase
{
    uniform RWTexture3D<Element> resourceView,

    Element doTest(uint3 coord)
    {
        // Read from resourceView
        Element elem = resourceView[coord];

        // Write to resourceView (if possible)
        resourceView[coord] = Element(1);

        return elem;
    }
}

struct UAV_2D : ITestCase
{
    uniform RWTexture2D<Element> resourceView,

    Element doTest(uint3 coord)
    {
        // Read from resourceView
        Element elem = resourceView[coord.xy];

        // Write to resourceView (if possible)
        resourceView[coord.xy] = Element(1);

        return elem;
    }
}

struct SRV_3D : ITestCase
{
    uniform Texture3D<Element> resourceView,

    Element doTest(uint3 coord)
    {
        // Read from resourceView
        Element elem = resourceView[coord];

        return elem;
    }
}

// Copy the contents of "src" into "dst". These are for 2D textures containing UINT data.
[shader("compute")]
[numthreads(1,1,1)]
void resourceViewTest<T:ITestCase>(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform uint width,
    uniform uint height,
    uniform uint depth,
    uniform T testCase,
    uniform RWStructuredBuffer<Element> testResults)
{
    uint tid = sv_dispatchThreadID.x;
    
    uint tmp = tid;
    uint x = tmp % width; tmp /= width;
    uint y = tmp % height; tmp /= height;
    uint z = tmp;

    uint3 coord = uint3(x, y, z);
    Element elem = testCase.doTest(coord);

    // Write something to testResults
    testResults[tid] = elem;
}
#endif
